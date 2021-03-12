"""
GatedConv as in gatedconv.py (where bond is represented as node), but bond as edge graph.
"""

from typing import Callable, Dict, Union

import dgl
import torch
from dgl import function as fn
from torch import nn

from rxnrep.model.utils import MLP


class GatedGCNConv(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_fc_layers: int = 1,
        batch_norm: bool = True,
        activation: Callable = nn.ReLU(),
        residual: bool = False,
        dropout: Union[float, None] = None,
    ):
        super(GatedGCNConv, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.residual = residual

        if input_dim != output_dim:
            self.residual = False

        # A, B, ... I are phi_1, phi_2, ..., phi_9 in the BonDNet paper
        hidden = [output_dim] * (num_fc_layers - 1)
        self.A = MLP(input_dim, hidden, activation=activation, out_size=output_dim)
        self.B = MLP(input_dim, hidden, activation=activation, out_size=output_dim)
        self.C = MLP(input_dim, hidden, activation=activation, out_size=output_dim)
        self.D = MLP(input_dim, hidden, activation=activation, out_size=output_dim)
        self.E = MLP(input_dim, hidden, activation=activation, out_size=output_dim)
        self.F = MLP(input_dim, hidden, activation=activation, out_size=output_dim)
        self.G = MLP(output_dim, hidden, activation=activation, out_size=output_dim)
        self.H = MLP(output_dim, hidden, activation=activation, out_size=output_dim)
        self.I = MLP(input_dim, hidden, activation=activation, out_size=output_dim)

        if self.batch_norm:
            self.bn_node_h = nn.BatchNorm1d(output_dim)
            self.bn_node_e = nn.BatchNorm1d(output_dim)
            self.bn_node_u = nn.BatchNorm1d(output_dim)

        delta = 1e-3
        if dropout is None or dropout < delta:
            self.dropout = nn.Identity()
        else:
            self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        g: dgl.DGLGraph,
        feats: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            g: the graphm bond as edges
            feats: {name, feats}. Atom, bond, and global features.

        Returns:
            updated features.
        """

        g = g.local_var()

        h = feats["atom"]
        e = feats["bond"]
        u = feats["global"]

        # for residual connection
        h_in = h
        e_in = e
        u_in = u

        #
        # update bond feature e
        #
        g.nodes["atom"].data.update({"Ah": self.A(h)})
        g.edges["bond"].data.update({"Be": self.B(e)})
        g.nodes["global"].data.update({"Cu": self.C(u)})

        # step 1
        # global feats to atom node (a simpy copy, sum operates on 1 tensor,
        # since we have only one global nodes connected to each atom node)
        g.update_all(fn.copy_u("Cu", "m"), fn.sum("m", "Cu"), etype="g2a")

        # step 2, aggregate edge features
        # sum of:
        # src and dst atom feats,
        # edge feats,
        # and global feats (already stored in atom nodes, here we only grab it from src)
        g.apply_edges(
            lambda edges: {
                "e": edges.src["Ah"]
                + edges.dst["Ah"]
                + edges.data["Be"]
                + edges.src["Cu"],
            },
            etype="bond",
        )
        e = g.edges["bond"].data["e"]

        # del for memory efficiency
        del g.nodes["atom"].data["Ah"]
        del g.edges["bond"].data["Be"]
        del g.nodes["global"].data["Cu"]

        if self.batch_norm:
            e = self.bn_node_e(e)
        e = self.activation(e)
        if self.residual:
            e = e_in + e

        #
        # update atom feature h
        #

        # step 1
        # edge feats to atom nodes: sum_j e_ij [Had] Eh_j
        g.nodes["atom"].data.update({"Eh": self.E(h)})
        g.edges["bond"].data["e"] = e
        g.update_all(atom_message_fn, atom_reduce_fn, etype="bond")
        try:
            h1 = g.nodes["atom"].data.pop("h1")
        except KeyError:
            # This only happens when there is no edges (e.g. single atom molecule H+).
            # Will not happen when the single atom molecule is batched with other
            # molecules. When batched, the batched graph has edges; thus `atom_reduce_fn`
            # will be called, and `h1` for the single atom molecule is initialized to a
            # zero tensor by dgl.
            h1 = 0.0

        # step 2
        # global feats to atom node (a simpy copy, sum operates on 1 tensor,
        # since we have only one global nodes connected to each atom node)
        g.nodes["global"].data.update({"Fu": self.F(u)})
        g.update_all(fn.copy_u("Fu", "m"), fn.sum("m", "Fu"), etype="g2a")
        h2 = g.nodes["atom"].data.pop("Fu")

        h = self.D(h) + h1 + h2

        # del for memory efficiency
        del g.nodes["atom"].data["Eh"]

        if self.batch_norm:
            h = self.bn_node_h(h)
        h = self.activation(h)
        if self.residual:
            h = h_in + h

        #
        # update global feature u
        #

        # Get the number of bonds of each atom (i.e. in degrees)
        # For single atom molecule, set the value from 0 to 1; we do this to avoid
        # divide by 0 error in `global_reduce_fn` below.
        degrees = g.in_degrees(etype="bond").reshape(-1, 1)
        degrees[degrees < 1] = 1

        g.nodes["atom"].data.update({"Gh": self.G(h), "degrees": degrees})
        g.edges["bond"].data.update({"He": self.H(e)})
        g.nodes["global"].data.update({"Iu": self.I(u)})

        # step 1
        # edge feats to atom nodes
        # No need to do the try except block as above for h1, since we use dgl built-in
        # functions here and fn.sum will always be called because of operation fushion?
        # But still, the feature He_sum for single atom molecule is initialized to a
        # zero tensor.
        g.update_all(fn.copy_e("He", "m"), fn.sum("m", "He_sum"), etype="bond")

        # step 2
        # aggregate global feats
        g.update_all(global_message_fn, global_reduce_fn, etype="a2g")
        u = g.nodes["global"].data.pop("u")

        if self.batch_norm:
            # do not apply batch norm if it there is only one graph and it is in
            # training mode, BN complains about it
            if u.shape[0] <= 1 and self.training:
                pass
            else:
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


def atom_message_fn(edges):
    return {
        "Eh_j": edges.src["Eh"],
        "sigma_eij": torch.sigmoid(edges.data["e"]),
    }


def atom_reduce_fn(nodes):
    Eh_j = nodes.mailbox["Eh_j"]
    sigma_eij = nodes.mailbox["sigma_eij"]

    # (sum_j eta_ij * Ehj)/(sum_j' eta_ij') <= dense attention
    h1 = torch.sum(sigma_eij * Eh_j, dim=1) / (torch.sum(sigma_eij, dim=1) + 1e-6)

    return {"h1": h1}


def global_message_fn(edges):
    return {
        "Gh": edges.src["Gh"],
        "He_sum": edges.src["He_sum"],
        "degrees": edges.src["degrees"],  # in degrees: number of bonds of each atom
    }


def global_reduce_fn(nodes):
    Gh = nodes.mailbox["Gh"]
    Iu = nodes.data["Iu"]
    He_sum = nodes.mailbox["He_sum"]
    degrees = nodes.mailbox["degrees"]  # in degrees: number of bonds of each atom

    # mean of edge features
    # He_sum is the sum of bond feats placed on atom nodes. We aggregating to global
    # nodes, each bond feature will be presented twice. However, we do NOT need to
    # divide it by 2, since here we use mean aggregation and the double counting is
    # already taken care of by the `torch.sum(degrees, dim=1)`.
    #
    # Also note that, for single atom molecule, degree is set to 1 (see above),
    # and He_sum is a zero tensors (set by dgl, see above). Then He_mean is a zero
    # tensor--what we wanted. Manually setting degree to 1 does not change the results.
    # But if we do not do He_mean and Gh_mean + He_mean + Iu, should be be careful
    # about whether this still holds as well.
    He_mean = torch.sum(He_sum, dim=1) / torch.sum(degrees, dim=1)
    Gh_mean = torch.mean(Gh, dim=1)

    return {"u": Gh_mean + He_mean + Iu}
