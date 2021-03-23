"""
Readout (pooling) layers.
"""
from typing import Any, Dict, List

import dgl
import numpy as np
import torch
from dgl.ops import segment_reduce, segment_softmax
from torch import nn


class Pooling(nn.Module):
    """
    Reaction feature pooling.

    Readout reaction features, one 1D tensor for each reaction.

    Supported methods:
    set2set: set2set for atom/bond features and then concatenate atom, bond, and global
        features.
    sum_cat_all: sum all atom, bond features (separately), and then concatenate atom,
        bond and global features.
    sum_cat_center: sum atoms, bonds features in reaction center, and then concatenate
        atom, bond and global features.
    global_only: only return global features

    Args:
        in_size: input feature size, i.e. atom/bond/global feature sizes
    """

    def __init__(
        self,
        in_size: int,
        pooling_method: str,
        pooling_kwargs: Dict[str, Any],
        has_global_feats: bool = True,
    ):
        super().__init__()

        self.pooling_method = pooling_method
        self.has_global_feats = has_global_feats

        if pooling_method == "set2set":
            if pooling_kwargs is None:
                num_iterations = 6
                num_layers = 3
            else:
                num_iterations = pooling_kwargs["set2set_num_iterations"]
                num_layers = pooling_kwargs["set2set_num_layers"]

            self.set2set_atom = Set2Set(
                input_dim=in_size, n_iters=num_iterations, n_layers=num_layers
            )
            self.set2set_bond = Set2Set(
                input_dim=in_size, n_iters=num_iterations, n_layers=num_layers
            )

            if has_global_feats:
                pooling_outsize = in_size * 5
            else:
                pooling_outsize = in_size * 4

        elif pooling_method in ["sum_cat_all", "sum_cat_center"]:
            if has_global_feats:
                pooling_outsize = in_size * 3
            else:
                pooling_outsize = in_size * 2

        elif pooling_method == "global_only":
            pooling_outsize = in_size

        elif pooling_method == "hop_distance":
            if pooling_kwargs is None:
                raise RuntimeError(
                    "`max_hop_distance` should be provided as `pooling_kwargs` to use "
                    "`hop_distance_pool`"
                )
            else:
                max_hop_distance = pooling_kwargs["max_hop_distance"]
                self.hop_dist_pool = HopDistancePooling(max_hop=max_hop_distance)

            pooling_outsize = in_size * 3

        else:
            raise ValueError(f"Unsupported pooling method `{pooling_method}`")

        self.reaction_feats_size = pooling_outsize

    def forward(
        self,
        molecule_graphs: dgl.DGLGraph,
        reaction_graphs: dgl.DGLGraph,
        feats: Dict[str, torch.Tensor],
        metadata: Dict[str, List[int]],
    ) -> torch.Tensor:
        """
        Returns:
            Reaction features, 2D tensor of shape (B, D), where B is batch size and D
            is reaction feature size.
        """

        # readout reaction features, a 1D tensor for each reaction
        if self.pooling_method == "set2set":
            device = feats["atom"].device
            atom_sizes = torch.as_tensor(metadata["num_atoms"], device=device)
            bond_sizes = torch.as_tensor(metadata["num_bonds"], device=device)

            atom_feats = feats["atom"]
            bond_feats = feats["bond"][::2]  # each bond has two edges, we select one

            rxn_feats_atom = self.set2set_atom(atom_feats, atom_sizes)
            rxn_feats_bond = self.set2set_bond(bond_feats, bond_sizes)

            if self.has_global_feats:
                return torch.cat(
                    [rxn_feats_atom, rxn_feats_bond, feats["global"]], dim=-1
                )
            else:
                return torch.cat([rxn_feats_atom, rxn_feats_bond], dim=-1)

        elif self.pooling_method == "sum_cat_all":
            device = feats["atom"].device
            atom_sizes = torch.as_tensor(metadata["num_atoms"], device=device)
            bond_sizes = torch.as_tensor(metadata["num_bonds"], device=device)

            atom_feats = feats["atom"]
            bond_feats = feats["bond"][::2]  # each bond has two edges, we select one

            rxn_feats_atom = segment_reduce(atom_sizes, atom_feats, reducer="sum")
            rxn_feats_bond = segment_reduce(bond_sizes, bond_feats, reducer="sum")

            if self.has_global_feats:
                return torch.cat(
                    [rxn_feats_atom, rxn_feats_bond, feats["global"]], dim=-1
                )
            else:
                return torch.cat([rxn_feats_atom, rxn_feats_bond], dim=-1)

        elif self.pooling_method == "sum_cat_center":
            atom_feats = feats["atom"]
            bond_feats = feats["bond"][::2]  # each bond has two edges, we select one

            # select feats of atoms/bonds in center
            aic = np.concatenate(metadata["atoms_in_reaction_center"]).tolist()
            bic = np.concatenate(metadata["bonds_in_reaction_center"]).tolist()
            atom_feats = atom_feats[aic]
            bond_feats = bond_feats[bic]

            # number of atoms/bonds in reaction center
            device = feats["atom"].device
            atom_sizes = torch.as_tensor(
                [sum(i) for i in metadata["atoms_in_reaction_center"]], device=device
            )
            bond_sizes = torch.as_tensor(
                [sum(i) for i in metadata["bonds_in_reaction_center"]], device=device
            )

            rxn_feats_atom = segment_reduce(atom_sizes, atom_feats, reducer="sum")
            rxn_feats_bond = segment_reduce(bond_sizes, bond_feats, reducer="sum")

            if self.has_global_feats:
                return torch.cat(
                    [rxn_feats_atom, rxn_feats_bond, feats["global"]], dim=-1
                )
            else:
                return torch.cat([rxn_feats_atom, rxn_feats_bond], dim=-1)

        elif self.pooling_method == "hop_distance":
            hop_dist = {
                "atom": metadata["atom_hop_dist"],
                "bond": metadata["bond_hop_dist"],
            }
            reaction_feats = self.hop_dist_pool(reaction_graphs, feats, hop_dist)

        elif self.pooling_method == "global_only":
            reaction_feats = feats["global"]

        else:
            raise ValueError(f"Unsupported pooling method `{self.pooling_method}`")

        return reaction_feats


class Set2Set(nn.Module):
    r"""
    Compute set2set for features (either node or edge features) of a batch of graph
    without requiring the batched graph.


    For each individual graph in the batch, set2set computes

    .. math::
        q_t &= \mathrm{LSTM} (q^*_{t-1})

        \alpha_{i,t} &= \mathrm{softmax}(x_i \cdot q_t)

        r_t &= \sum_{i=1}^N \alpha_{i,t} x_i

        q^*_t &= q_t \Vert r_t

    for this graph.

    Args:
        input_dim: The size of each input sample.
        n_iters: The number of iterations.
        n_layers: The number of recurrent layers.
    """

    def __init__(self, input_dim: int, n_iters: int, n_layers: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = 2 * input_dim
        self.n_iters = n_iters
        self.n_layers = n_layers
        self.lstm = torch.nn.LSTM(self.output_dim, self.input_dim, n_layers)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        self.lstm.reset_parameters()

    def forward(self, feat: torch.Tensor, sizes=torch.Tensor) -> torch.Tensor:
        """
        Compute set2set pooling.

        Args:
            feat: feature tensor of shape (N, D) where N is the total number of features,
                and D is the feature dimension.
            sizes: 1D tensor (shape (B,)) of the size of the features for each graph.
                sum(sizes) should be equal to D.

        Returns:
            Aggregated output feature with shape (B, D), where B is the batch size
            (i.e. number of graphs) and D means the size of features.
        """
        batch_size = len(sizes)

        h = (
            feat.new_zeros((self.n_layers, batch_size, self.input_dim)),
            feat.new_zeros((self.n_layers, batch_size, self.input_dim)),
        )

        q_star = feat.new_zeros(batch_size, self.output_dim)

        for _ in range(self.n_iters):
            q, h = self.lstm(q_star.unsqueeze(0), h)
            q = q.view(batch_size, self.input_dim)
            e = (feat * torch.repeat_interleave(q, sizes, dim=0)).sum(
                dim=-1, keepdim=True
            )
            alpha = segment_softmax(sizes, e)
            r = feat * alpha
            readout = segment_reduce(sizes, r, reducer="sum")
            q_star = torch.cat([q, readout], dim=-1)

        return q_star


class HopDistancePooling(nn.Module):
    """
    Pooling atom/bond features based on their distance from reaction center.

    The pooled feature is a weighted sum of the features of all atoms/bonds in the
    reaction. The weight is based on a weight function.

    For example, if the cosine function is used, atoms/bonds in the reaction center will
    have a weight of 1.0 and the weight decays to 0 for atoms/bonds of `max_hop`
    distance from the reaction center.

    """

    def __init__(self, max_hop: int, weight_fn: str = "cos"):
        super(HopDistancePooling, self).__init__()

        self.max_hop = max_hop
        self.weight_fn = weight_fn

    def forward(
        self,
        graph: dgl.DGLGraph,
        feats: Dict[str, torch.Tensor],
        hop_distance: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            graph: reaction graph
            feats: node features with node type (atom/bond/global) as key and the
                corresponding features as value. Each tensor is of shape (N, D)
                where N is the number of nodes of the corresponding node type, and D is
                the feature size (D could be different for different node features).
            hop_distance: Atom/bond hop distance of from reaction center. Each tensor
                is of shape (N,), where N is the number of nodes of the corresponding
                node type.
        Returns:
            2D tensor of shape (N, D1+D2+D3), where N is the number of reactions in
                the batch, and D1, D2, and D3 are atom, bond, and global feature sizes,
                respectively.
        """
        atom_readout = self.readout(graph, feats["atom"], hop_distance["atom"], "atom")
        bond_readout = self.readout(graph, feats["bond"], hop_distance["bond"], "bond")
        global_readout = feats["global"]

        res = torch.cat((atom_readout, bond_readout, global_readout), dim=-1)

        return res

    def readout(self, graph, feats: torch.Tensor, hop_dist: torch.Tensor, ntype: str):

        # Set atom/bond in the reaction center to have hop distance 0
        # We need this because in `grapher.get_atom_distance_to_reaction_center()`
        # atoms in added bond has hop distance of `max_hop+1`, and atom is in both
        # broken bond and added bond have hop distance `max_hop+2`.
        # and `grapher.get_bond_distance_to_reaction_center()`
        # Similarly, added bonds has a hop distance of `max_hop+1`.
        hop_dist[hop_dist > self.max_hop] = 0

        # convert hop dist to angle such that host dist 0 have angle 0, and hop
        # distance self.max_hop has angle pi/2

        pi_over_2 = float(np.pi / 2.0)
        angle = pi_over_2 * hop_dist / self.max_hop

        weight = torch.cos(angle)
        graph.nodes[ntype].data["r"] = feats * weight.view(-1, 1)
        weighted_sum = dgl.sum_nodes(graph, "r", ntype=ntype)

        return weighted_sum
