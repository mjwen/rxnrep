"""
The directed MPNN as used in:
Analyzing Learned Molecular Representations for Property Prediction,
J. Chem. Inf. Model. 2019, 59, 3370−3388.
"""


from typing import Callable, Dict, List, Union

import dgl
import torch
from dgl import function as fn
from dgl.ops import segment_reduce
from torch import nn

from rxnrep.model.utils import MLP, get_activation, get_dropout


class DMPNNConvAtomMessage(nn.Module):
    def __init__(
        self,
        atom_feat_dim: int,
        bond_feat_dim: int,
        output_dim: int,
        depth: int = 6,
        activation: Callable = nn.ReLU(),
        dropout: Union[float, None] = None,
    ):
        super().__init__()
        self.depth = depth

        # input
        self.W_i = nn.Linear(atom_feat_dim, output_dim, bias=False)

        # shared weight matrix across depths
        self.W_h = nn.Linear(bond_feat_dim + output_dim, output_dim, bias=False)

        self.W_o = nn.Linear(atom_feat_dim + output_dim, output_dim)

        self.activation = get_activation(activation)
        self.dropout = get_dropout(dropout)

    def forward(self, g: dgl.DGLGraph, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        """

        Args:
            g: a batch of molecule graphs

        Returns:
            Updated atom feats. The model only update atom features, not bond features.
        """

        g = g.local_var()

        h = feats["atom"]
        e = feats["bond"]
        g.edges["bond"].data["e"] = e

        input = self.W_i(h)
        message = self.activation(input)

        #
        # message passing
        # sum_j [h_j || e_ij]
        #
        for depth in range(self.depth - 1):
            g.nodes["atom"].data["h"] = message

            g.update_all(
                lambda edges: {
                    "m": torch.cat((edges.src["h"], edges.data["e"]), dim=-1)
                },
                fn.sum("m", "message"),
                etype="bond",
            )
            message = g.ndoes["atom"].data.pop("message")
            message = self.dropout(self.activation(input + self.W_h(message)))

        #
        # output
        # sum_j (h_j)
        # [h || sum_hj]
        g.nodes["atom"].data["h"] = message
        g.update_all(fn.copy_u("h", "m"), fn.sum("m", "sum_hj"), etype="bond")
        sum_hj = g.nodes["atom"].data["sum_hj"]

        h = torch.cat((h, sum_hj), dim=-1)
        h = self.dropout(self.activation(self.W_o(h)))

        return h


class DMPNNConvBondMessage(nn.Module):
    def __init__(
        self,
        atom_feat_dim: int,
        bond_feat_dim: int,
        output_dim: int,
        depth: int = 6,
        activation: Callable = nn.ReLU(),
        dropout: Union[float, None] = None,
    ):
        super().__init__()

        self.output_dim = output_dim
        self.depth = depth

        # Input
        self.W_i = nn.Linear(atom_feat_dim + bond_feat_dim, output_dim, bias=False)

        # Shared weight matrix across depths (default)
        self.W_h = nn.Linear(output_dim, output_dim, bias=False)

        self.W_o = nn.Linear(atom_feat_dim + output_dim, output_dim)

        self.activation = get_activation(activation)
        self.dropout = get_dropout(dropout)

    def forward(self, g: dgl.DGLGraph, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        """

        Args:
            g: a batch of molecule graphs

        Returns:
            Updated atom feats. The model only update atom features, not bond features.
        """

        g = g.local_var()

        h = feats["atom"]
        e = feats["bond"]

        #
        # input
        # [h || e]
        g.nodes["atom"].data["h"] = h
        g.edges["bond"].data["e"] = e
        g.apply_edges(
            lambda edges: {"he": torch.cat((edges.src["h"], edges.data["e"]), dim=-1)},
            etype="bond",
        )
        he = g.edges["bond"].data["he"]

        input = self.W_i(he)
        message = self.activation(input)

        #
        # message passing
        #

        # swap = [1,0,3,2,5,4, ...]
        swap = [i + 1 if i % 2 == 0 else i - 1 for i in range(len(message))]

        # m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1)] - m(a2 -> a1)
        # message      a_message = sum(nei_a_message)      rev_message
        for depth in range(self.depth - 1):

            g.edges["bond"].data["e"] = message

            # e_sum = sum_{a0 \in nei(a1)} m(a0 -> a1)
            g.update_all(fn.copy_e("e", "m"), fn.sum("m", "e_sum"), etype="bond")

            # e_sum - m(a2 -> a1)
            g.apply_edges(fn.v_sub_e("e_sum", "e", "e_sum_no_a2"), etype="bond")
            message = g.edges["bond"].data.pop("e_sum_no_a2")

            # We use v_sub_e (target node subtracting edge), so the result will be
            # placed on edge from a2 to a1, but we actually want it be placed on edge
            # a1->a2. So we swap the message to make it happen (recall that 2i and 2i+1
            # are edges for the same atom).
            message = message[swap]

            message = self.dropout(self.activation(input + self.W_h(message)))

        #
        # prepare output
        #
        g.edges["bond"].data["e"] = message
        g.update_all(fn.copy_e("e", "m"), fn.sum("m", "sum_ej"), etype="bond")
        sum_ej = g.nodes["atom"].data["sum_ej"]

        h = torch.cat([h, sum_ej], dim=-1)
        h = self.dropout(self.activation(self.W_o(h)))

        return h


class DMPNNEncoder(nn.Module):
    def __init__(
        self,
        atom_feat_dim: int,
        bond_feat_dim: int,
        output_dim: int,
        depth: int = 5,
        activation: Callable = nn.ReLU(),
        dropout: Union[float, None] = None,
        pooling_method="sum",
        conv="BondMessage",
    ):
        super().__init__()
        self.pooling_method = pooling_method

        if conv == "BondMessage":
            self.conv_layer = DMPNNConvBondMessage(
                atom_feat_dim=atom_feat_dim,
                bond_feat_dim=bond_feat_dim,
                output_dim=output_dim,
                depth=depth,
                activation=activation,
                dropout=dropout,
            )
        elif conv == "AtomMessage":
            self.conv_layer = DMPNNConvAtomMessage(
                atom_feat_dim=atom_feat_dim,
                bond_feat_dim=bond_feat_dim,
                output_dim=output_dim,
                depth=depth,
                activation=activation,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Not supported conv {conv}")

        # the single layer in chemprop MPNDiff
        self.mlp = nn.Linear(output_dim, output_dim, bias=False)
        self.activation = get_activation(activation)
        self.dropout = get_dropout(dropout)

        self.reaction_feats_size = output_dim

    def forward(
        self,
        molecule_graphs: dgl.DGLGraph,
        reaction_graphs: dgl.DGLGraph,
        feats: Dict[str, torch.Tensor],
        metadata: Dict[str, List[int]],
    ):
        """
        Get a vector representation for each reaction.

        Args:
            molecule_graphs:
            reaction_graphs:
            feats:
            metadata:

        Returns:
            2D tensor of shape (B, D), where B is the batch size and D is the feature
            dimension, i.e. output_dim.
        """

        atom_feats = self.conv_layer(molecule_graphs, feats)

        # same number of atom nodes in reactants and products
        size = len(atom_feats) // 2
        # we can do the below to lines because in the collate fn of dataset, all products
        # graphs are appended to reactants graphs
        reactant_atom_feats = atom_feats[:size]
        product_atom_feats = atom_feats[size:]

        diff_feats = product_atom_feats - reactant_atom_feats

        diff_feats = self.dropout(self.activation(self.mlp(diff_feats)))

        #
        # pooling
        #
        num_atoms = reaction_graphs.batch_num_nodes("atom")
        rxn_feats = segment_reduce(num_atoms, diff_feats, reducer=self.pooling_method)

        return None, rxn_feats


class DMPNNModel(DMPNNEncoder):
    def __init__(
        self,
        atom_feat_dim: int,
        bond_feat_dim: int,
        output_dim: int,
        depth: int = 6,
        activation: Callable = nn.ReLU(),
        dropout: Union[float, None] = None,
        # pooling
        pooling_method="sum",
        # reaction energy decoder
        reaction_energy_decoder_hidden_layer_sizes=None,
        reaction_energy_decoder_activation=None,
        # activation energy decoder
        activation_energy_decoder_hidden_layer_sizes=None,
        activation_energy_decoder_activation=None,
    ):
        super().__init__(
            atom_feat_dim=atom_feat_dim,
            bond_feat_dim=bond_feat_dim,
            output_dim=output_dim,
            depth=depth,
            activation=activation,
            dropout=dropout,
            pooling_method=pooling_method,
        )

        # ========== reaction level decoder ==========

        # reaction energy decoder
        if reaction_energy_decoder_hidden_layer_sizes:
            self.reaction_energy_decoder = MLP(
                in_size=self.reaction_feats_size,
                hidden_sizes=reaction_energy_decoder_hidden_layer_sizes,
                activation=reaction_energy_decoder_activation,
                out_size=1,
            )
        else:
            self.reaction_energy_decoder = None

        # activation energy decoder
        if activation_energy_decoder_hidden_layer_sizes:
            self.activation_energy_decoder = MLP(
                in_size=self.reaction_feats_size,
                hidden_sizes=activation_energy_decoder_hidden_layer_sizes,
                activation=activation_energy_decoder_activation,
                out_size=1,
            )
        else:
            self.activation_energy_decoder = None

    def decode(
        self,
        feats: torch.Tensor,
        reaction_feats: torch.Tensor,
        metadata: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        # reaction decoder
        reaction_energy = (
            None
            if self.reaction_energy_decoder is None
            else self.reaction_energy_decoder(reaction_feats)
        )
        activation_energy = (
            None
            if self.activation_energy_decoder is None
            else self.activation_energy_decoder(reaction_feats)
        )

        # predictions
        predictions = {
            "reaction_energy": reaction_energy,
            "activation_energy": activation_energy,
        }

        return predictions
