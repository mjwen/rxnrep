from typing import Callable, Dict, Union

import dgl
import torch
from dgl import function as fn
from torch import nn

from rxnrep.model.utils import FCNN


class GatedGCNConv(nn.Module):
    """
    Gated graph convolutional layer to update molecular features.

    It update bond, atom, and global features in sequence. See the BonDNet paper for
    details. This is a direct extension of the Residual Gated Graph ConvNets
    (https://arxiv.org/abs/1711.07553) by adding global features.

    Args:
        input_dim: input feature dimension
        output_dim: output feature dimension
        num_fc_layers: number of NN layers to transform input to output. In `Residual
            Gated Graph ConvNets` the number of layers is set to 1. Here we make it a
            variable to accept any number of layers.
        graph_norm: whether to apply the graph norm proposed in
            Benchmarking Graph Neural Networks (https://arxiv.org/abs/2003.00982)
        batch_norm: whether to apply batch normalization
        activation: activation function
        residual: whether to add residual connection as in the ResNet:
            Deep Residual Learning for Image Recognition (https://arxiv.org/abs/1512.03385)
        dropout: dropout ratio. Note, dropout is applied after residual connection.
            If `None`, do not apply dropout.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_fc_layers: int = 1,
        graph_norm: bool = False,
        batch_norm: bool = True,
        activation: Callable = nn.ReLU(),
        residual: bool = False,
        dropout: Union[float, None] = None,
    ):
        super(GatedGCNConv, self).__init__()
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.activation = activation
        self.residual = residual

        if input_dim != output_dim:
            self.residual = False

        out_sizes = [output_dim] * num_fc_layers
        acts = [activation] * (num_fc_layers - 1) + [nn.Identity()]
        use_bias = [True] * num_fc_layers

        # A, B, ... I are phi_1, phi_2, ..., phi_9 in the BonDNet paper
        self.A = FCNN(input_dim, out_sizes, acts, use_bias)
        self.B = FCNN(input_dim, out_sizes, acts, use_bias)
        self.C = FCNN(input_dim, out_sizes, acts, use_bias)
        self.D = FCNN(input_dim, out_sizes, acts, use_bias)
        self.E = FCNN(input_dim, out_sizes, acts, use_bias)
        self.F = FCNN(input_dim, out_sizes, acts, use_bias)
        self.G = FCNN(output_dim, out_sizes, acts, use_bias)
        self.H = FCNN(output_dim, out_sizes, acts, use_bias)
        self.I = FCNN(input_dim, out_sizes, acts, use_bias)

        if self.batch_norm:
            self.bn_node_h = nn.BatchNorm1d(output_dim)
            self.bn_node_e = nn.BatchNorm1d(output_dim)
            self.bn_node_u = nn.BatchNorm1d(output_dim)

        delta = 1e-3
        if dropout is None or dropout < delta:
            self.dropout = nn.Identity()
        else:
            self.dropout = nn.Dropout(dropout)

    @staticmethod
    def message_fn(edges):
        return {
            "Eh_sum": edges.src["Eh_sum"],
            "sigma_eij": torch.sigmoid(edges.src["e"]),
        }

    @staticmethod
    def reduce_fn(nodes):
        Eh_i = nodes.data["Eh"]
        Eh_sum = nodes.mailbox["Eh_sum"]
        sigma_eij = nodes.mailbox["sigma_eij"]

        # Eh_i is a 2D tensor (d0, d1)
        # d0: node batch dim
        # d1: feature dim
        #
        # Eh_sum and e are 3D tensors (d0, d1, d2).
        # d0: node batch dim, i.e. the messages for different node
        # d1: message dim, i.e. all messages for one node from other node
        # d2: feature dim
        shape = Eh_i.shape
        Eh_j = Eh_sum - Eh_i.view(shape[0], 1, shape[1])

        # (sum_j eta_ij * Ehj)/(sum_j' eta_ij') <= dense attention
        h = torch.sum(sigma_eij * Eh_j, dim=1) / (torch.sum(sigma_eij, dim=1) + 1e-6)

        return {"h": h}

    def forward(
        self,
        g: dgl.DGLGraph,
        feats: Dict[str, torch.Tensor],
        norm_atom: torch.Tensor = None,
        norm_bond: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            g: the graph
            feats: node features. Allowed node types are `atom`, `bond` and `global`.
            norm_atom: values used to normalize atom features as proposed in graph norm.
            norm_bond: values used to normalize bond features as proposed in graph norm.

        Returns:
            updated node features.
        """

        g = g.local_var()

        h = feats["atom"]
        e = feats["bond"]
        u = feats["global"]

        # for residual connection
        h_in = h
        e_in = e
        u_in = u

        g.nodes["atom"].data.update({"Ah": self.A(h), "Eh": self.E(h)})
        g.nodes["global"].data.update({"Cu": self.C(u), "Fu": self.F(u)})

        # update bond feature e
        g.multi_update_all(
            {
                "a2b": (fn.copy_u("Ah", "m"), fn.sum("m", "e")),  # A * (h_i + h_j)
                "g2b": (fn.copy_u("Cu", "m"), fn.sum("m", "e")),  # C * u
            },
            "sum",
        )
        e = g.nodes["bond"].data["e"] + self.B(e)  # B * e_ij

        if self.graph_norm:
            e = e * norm_bond
        if self.batch_norm:
            e = self.bn_node_e(e)
        e = self.activation(e)
        if self.residual:
            e = e_in + e
        g.nodes["bond"].data["e"] = e

        # update atom feature h

        # We do this in a two step fashion: Eh_j -> bond node -> atom i node
        # To get e_ij [Had] Eh_j, we use the trick:
        # e_ij [Had] Eh_j = e_ij [Had] [(Eh_j + Eh_i) - Eh_i]
        # This is achieved in two steps:
        # step 1: Eh_sum = Eh_j + Eh_i
        # step 2: e_ij[Had] Eh_j = e_ij[Had][Eh_sum - Eh_i]

        g.update_all(fn.copy_u("Eh", "m"), fn.sum("m", "Eh_sum"), etype="a2b")  # step 1

        g.multi_update_all(
            {
                "b2a": (self.message_fn, self.reduce_fn),  # step 2
                "g2a": (fn.copy_u("Fu", "m"), fn.sum("m", "h")),  # F * u
            },
            "sum",
        )
        h = g.nodes["atom"].data["h"] + self.D(h)  # D * h_i

        if self.graph_norm:
            h = h * norm_atom
        if self.batch_norm:
            h = self.bn_node_h(h)
        h = self.activation(h)
        if self.residual:
            h = h_in + h
        g.nodes["atom"].data["h"] = h

        # update global feature u

        g.nodes["atom"].data.update({"Gh": self.G(h)})
        g.nodes["bond"].data.update({"He": self.H(e)})
        g.multi_update_all(
            {
                "a2g": (fn.copy_u("Gh", "m"), fn.mean("m", "u")),  # G * (mean_i h_i)
                "b2g": (fn.copy_u("He", "m"), fn.mean("m", "u")),  # H * (mean_ij e_ij)
            },
            "sum",
        )
        u = g.nodes["global"].data["u"] + self.I(u)  # I * u

        # do not apply batch norm if it there is only one graph
        if self.batch_norm and u.shape[0] > 1:
            u = self.bn_node_u(u)
        u = self.activation(u)
        if self.residual:
            u = u_in + u

        # dropout
        h = self.dropout(h)
        e = self.dropout(e)
        u = self.dropout(u)

        feats = {"atom": h, "bond": e, "global": u}

        return feats
