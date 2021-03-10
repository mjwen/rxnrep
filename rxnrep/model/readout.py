"""
Readout (pooling) layers.
"""
from typing import Dict, List, Optional, Tuple

import dgl
import numpy as np
import torch
from dgl import function as fn
from dgl.ops import segment_reduce, segment_softmax
from torch import nn

from rxnrep.model.utils import MLP


class ConcatenateMeanMax(nn.Module):
    """
    Concatenate the mean and max of features of a node type to another node type.

    Args:
        etypes: canonical edge types of a graph of which the features of node
            `u` are concatenated to the features of node `v`.
            For example: if `etypes = [('atom', 'a2b', 'bond'), ('global','g2b', 'bond')]`
            then the mean and max of the features of `atom` as well as  `global` are
            concatenated to the features of `bond`.
    """

    def __init__(self, etypes: List[Tuple[str, str, str]]):
        super(ConcatenateMeanMax, self).__init__()
        self.etypes = etypes

    def forward(
        self, graph: dgl.DGLGraph, feats: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            graph: the graph
            feats: node features with node type as key and the corresponding
                features as value. Each tensor is of shape (N, D) where N is the number
                of nodes of the corresponding node type, and D is the feature size.

        Returns:
            updated node features. Each tensor is of shape (N, D) where N is the number
            of nodes of the corresponding node type, and D is the feature size.

        """
        graph = graph.local_var()

        # assign data
        for nt, ft in feats.items():
            graph.nodes[nt].data.update({"ft": ft})

        for et in self.etypes:
            # option 1
            graph[et].update_all(fn.copy_u("ft", "m"), fn.mean("m", "mean"), etype=et)
            graph[et].update_all(fn.copy_u("ft", "m"), fn.max("m", "max"), etype=et)

            nt = et[2]
            graph.apply_nodes(self._concatenate_node_feat, ntype=nt)

            # copy update feature from new_ft to ft
            graph.nodes[nt].data.update({"ft": graph.nodes[nt].data["new_ft"]})

        return {nt: graph.nodes[nt].data["ft"] for nt in feats}

    @staticmethod
    def _concatenate_node_feat(nodes):
        data = nodes.data["ft"]
        mean = nodes.data["mean"]
        max = nodes.data["max"]
        concatenated = torch.cat((data, mean, max), dim=1)
        return {"new_ft": concatenated}


class ConcatenateMeanAbsDiff(nn.Module):
    """
    Concatenate the mean and max of features of a node type to another node type.

    This is very specific to the scheme that two atoms directed to bond. Others may fail.

    Args:
        etypes: canonical edge types of a graph of which the features of node `u`
            are concatenated to the features of node `v`.
            For example: if `etypes = [('atom', 'a2b', 'bond'), ('global','g2b', 'bond')]`
            then the mean and max of the features of `atom` and `global` are concatenated
            to the features of `bond`.
    """

    def __init__(self, etypes):
        super(ConcatenateMeanAbsDiff, self).__init__()
        self.etypes = etypes

    def forward(
        self, graph: dgl.DGLGraph, feats: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:

        """
        Args:
            graph: the graph
            feats: node features with node type as key and the corresponding
                features as value. Each tensor is of shape (N, D) where N is the number
                of nodes of the corresponding node type, and D is the feature size.

        Returns:
            updated node features. Each tensor is of shape (N, D) where N is the number
            of nodes of the corresponding node type, and D is the feature size.
        """
        graph = graph.local_var()

        # assign data
        for nt, ft in feats.items():
            graph.nodes[nt].data.update({"ft": ft})

        for et in self.etypes:
            graph[et].update_all(fn.copy_u("ft", "m"), self._concatenate_data, etype=et)

        return {nt: graph.nodes[nt].data["ft"] for nt in feats}

    @staticmethod
    def _concatenate_data(nodes):
        message = nodes.mailbox["m"]
        mean_v = torch.mean(message, dim=1)
        # NOTE this is very specific to the atom -> bond case
        # there are two elements along dim=1, since for each bond we have two atoms
        # directed to it
        abs_diff = torch.stack([torch.abs(x[0] - x[1]) for x in message])
        data = nodes.data["ft"]

        concatenated = torch.cat((data, mean_v, abs_diff), dim=1)
        return {"ft": concatenated}


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


class Set2SetThenCat(nn.Module):
    """
    Set2Set for nodes (separate for different node type) and then concatenate the
    features of different node types to create a representation of the graph.

    Note, this assume the the bond edge is bidirectionally, i.e. two bond edge features
    for one bond; thus, we only select one of the two exactly the same feature for a
    bond.

     Args:
        num_iters: number of LSTM iteration
        num_layers: number of LSTM layers
        in_feats: size of input features
        ntypes: node types to perform Set2Set, e.g. ['atom`].
        etypes: edge types to perform Set2Set, e.g. ['bond`].
        ntypes_direct_cat: node types to which not perform Set2Set, whose features are
            directly concatenated. e.g. ['global']
    """

    def __init__(
        self,
        num_iters: int,
        num_layers: int,
        in_feats: int,
        ntypes: List[str],
        etypes: List[str],
        ntypes_direct_cat: Optional[List[str]] = None,
    ):
        super(Set2SetThenCat, self).__init__()
        self.ntypes = ntypes
        self.etypes = etypes
        self.ntypes_direct_cat = ntypes_direct_cat

        self.node_layers = nn.ModuleDict()
        for t in ntypes:
            self.node_layers[t] = Set2Set(
                input_dim=in_feats, n_iters=num_iters, n_layers=num_layers
            )

        self.edge_layers = nn.ModuleDict()
        for t in etypes:
            self.edge_layers[t] = Set2Set(
                input_dim=in_feats, n_iters=num_iters, n_layers=num_layers
            )

    def forward(
        self, graph: dgl.DGLGraph, feats: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            graph: the graph
            feats: node features with node type as key and the corresponding
                features as value. Each tensor is of shape (N, D) where N is the number
                of nodes of the corresponding node type, and D is the feature size
                (D could be different for different node features).

        Returns:
            A tensor representation of the each graph, of shape
            (N, 2D_1+2D_2+ ... D_{m-1}, D_m),
            where N is the batch size (number of graphs), and D_1, D_2 ... are the
            feature sizes of the nodes to perform the set2set aggregation. The `2`
            shows up because set2set doubles the feature sizes. ... D_{m-1}, D_m are
            the feature sizes of the nodes not to perform set2set by direct concatenate.
        """
        rst = []
        for t in self.ntypes:
            ft = self.node_layers[t](feats[t], graph.batch_num_nodes(t))
            rst.append(ft)

        for t in self.etypes:
            ft = feats[t]
            ft = ft[::2]  # each bond has two edges, we select one
            sizes = graph.batch_num_edges(t) // 2
            ft = self.edge_layers[t](ft, sizes)
            rst.append(ft)

        if self.ntypes_direct_cat is not None:
            for nt in self.ntypes_direct_cat:
                rst.append(feats[nt])

        res = torch.cat(rst, dim=-1)

        return res


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


class CompressingNN(nn.Module):
    """
    A fully connected NN with (expecting) fewer number of nodes in later layers.

    Used as a way to compressing information in the encoder.

    Note, we use bias for all layers, since this will be internal layers, not final
    prediction head layers.
    """

    def __init__(self, in_size: int, hidden_sizes: List[int], activation="ReLU"):
        super().__init__()
        acts = [activation] * len(hidden_sizes)
        use_bias = [True] * len(hidden_sizes)

        self.layers = MLP(in_size, hidden_sizes, acts, use_bias)

    def forward(self, feats):
        return self.layers(feats)
