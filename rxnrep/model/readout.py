"""
Readout (pool) layers.
"""
from typing import Any, Callable, Dict, List, Tuple

import dgl
import numpy as np
import torch
from dgl.ops import segment_reduce, segment_softmax
from torch import nn


class BasePooling(nn.Module):
    """
    Args:
        in_size: input feature size, i.e. atom/bond/global feature sizes
        pool_atom/bond/global_feats: whether to include atom/bond/global features in the
            final representation.
        reducer: method to aggregate feats
    """

    def __init__(
        self,
        in_size: int,
        pool_atom_feats: bool = True,
        pool_bond_feats: bool = True,
        pool_global_feats: bool = True,
        reducer="sum",
    ):

        super().__init__()

        if not any([pool_atom_feats, pool_bond_feats, pool_global_feats]):
            raise ValueError(
                "Expect one of atom/bond/global pool to be true; got False for all"
            )
        if reducer not in ["sum", "mean"]:
            raise ValueError(f"Expect reducer be sum or mean; got {reducer}")

        self.in_size = in_size
        self.reducer = reducer
        self.out_size = 0

        if pool_atom_feats:
            self.pool_atom, size = self.init_atom_pool_method()
            self.out_size += size
        else:
            self.pool_atom = None

        if pool_bond_feats:
            self.pool_bond, size = self.init_bond_pool_method()
            self.out_size += size
        else:
            self.pool_bond = None

        if pool_global_feats:
            self.pool_global, size = self.init_global_pool_method()
            self.out_size += size
        else:
            self.pool_global = None

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
            is output size (self.out_size), typically reaction feature size.
        """
        pooled = []
        if self.pool_atom:
            atom_feats, atom_sizes = self.get_atom_feats_and_sizes(feats, metadata)
            atom_feats_pooled = self.pool_atom(atom_feats, atom_sizes)
            pooled.append(atom_feats_pooled)

        if self.pool_bond:
            bond_feats, bond_sizes = self.get_bond_feats_and_sizes(feats, metadata)
            bond_feats_pooled = self.pool_bond(bond_feats, bond_sizes)
            pooled.append(bond_feats_pooled)

        if self.pool_global:
            global_feats_pooled = feats["global"]
            pooled.append(global_feats_pooled)

        # concatenate atom, bond, global feats
        return torch.cat(pooled, dim=-1)

    @staticmethod
    def get_atom_feats_and_sizes(feats, metadata):
        atom_feats = feats["atom"]
        atom_sizes = torch.as_tensor(metadata["num_atoms"], device=atom_feats.device)

        return atom_feats, atom_sizes

    @staticmethod
    def get_bond_feats_and_sizes(feats, metadata):
        bond_feats = feats["bond"][::2]  # each bond has two edges, we select one
        bond_sizes = torch.as_tensor(metadata["num_bonds"], device=bond_feats.device)

        return bond_feats, bond_sizes

    def init_atom_pool_method(self) -> Tuple[Callable, int]:
        """
        Returns:
            pool: a method to pool atom feats
            size: out size of pooled atom feats

        """
        raise NotImplementedError

    def init_bond_pool_method(self):
        """
        Returns:
            pool: a method to pool bond feats
            size: out size of pooled bond feats

        """
        raise NotImplementedError

    def init_global_pool_method(self):
        """
        Returns:
            pool: a method to pool bond feats
            size: out size of pooled bond feats

        """
        return True, self.in_size


class ReducePool(BasePooling):
    """
    Sum/mean reduce pool of all atom/bond/global features.
    """

    def init_atom_pool_method(self):
        def method(feats, sizes):
            return segment_reduce(sizes, feats, reducer=self.reducer)

        out_size = self.in_size

        return method, out_size

    def init_bond_pool_method(self):
        return self.init_atom_pool_method()


class CenterReducePool(BasePooling):
    """
    Sum/mean reduce pool of the features of atom/bond/global in the reaction center.
    """

    @staticmethod
    def get_atom_feats_and_sizes(feats, metadata):
        # atoms in center
        atom_feats = feats["atom"]
        aic = np.concatenate(metadata["atoms_in_reaction_center"]).tolist()
        atom_feats = atom_feats[aic]
        atom_sizes = torch.as_tensor(
            [sum(i) for i in metadata["atoms_in_reaction_center"]],
            device=atom_feats.device,
        )

        return atom_feats, atom_sizes

    @staticmethod
    def get_bond_feats_and_sizes(feats, metadata):
        # bonds in center
        bond_feats = feats["bond"][::2]  # each bond has two edges, we select one
        bic = np.concatenate(metadata["bonds_in_reaction_center"]).tolist()
        bond_feats = bond_feats[bic]
        bond_sizes = torch.as_tensor(
            [sum(i) for i in metadata["bonds_in_reaction_center"]],
            device=bond_feats.device,
        )
        return bond_feats, bond_sizes

    def init_atom_pool_method(self):
        def method(feats, sizes):
            return segment_reduce(sizes, feats, reducer=self.reducer)

        out_size = self.in_size

        return method, out_size

    def init_bond_pool_method(self):
        return self.init_atom_pool_method()


class AttentiveReducePool(BasePooling):
    """
    Attentive sum/mean reduce pool of all atom/bond/global features.
    """

    def __init__(
        self,
        in_size: int,
        pool_atom_feats: bool = True,
        pool_bond_feats: bool = True,
        pool_global_feats: bool = True,
        reducer="mean",
        activation: str = "LeakyReLU",
    ):
        self.activation = activation
        super().__init__(
            in_size, pool_atom_feats, pool_bond_feats, pool_global_feats, reducer
        )

    def init_atom_pool_method(self):
        method = AttentiveReduce(
            in_size=self.in_size, activation=self.activation, reducer=self.reducer
        )
        out_size = self.in_size

        return method, out_size

    def init_bond_pool_method(self):
        return self.init_atom_pool_method()

    def get_attention_score(
        self,
        molecule_graphs: dgl.DGLGraph,
        reaction_graphs: dgl.DGLGraph,
        feats: Dict[str, torch.Tensor],
        metadata: Dict[str, List[int]],
    ):
        """

        Args:
            molecule_graphs:
            reaction_graphs:
            feats:
            metadata:

        Returns:
            Attention score dict. {name: torch.Tensor}, where name is atom or bond.
            The tensor if of shape (N, D), where N is the total number of atoms/bonds
            in the batch and D is feature dimension.

        """
        attn_score = {}
        if self.pool_atom:
            atom_feats, atom_sizes = self.get_atom_feats_and_sizes(feats, metadata)
            atom_attn_score = self.pool_atom.get_attention_score(atom_feats, atom_sizes)
            attn_score["atom"] = atom_attn_score

        if self.pool_bond:
            bond_feats, bond_sizes = self.get_bond_feats_and_sizes(feats, metadata)
            bond_attn_score = self.pool_bond.get_attention_score(bond_feats, bond_sizes)
            attn_score["bond"] = bond_attn_score

        return attn_score


class Set2SetPool(BasePooling):
    """
    Set to set sum/mean reduce pool of all atom/bond/global features.
    """

    def __init__(
        self,
        in_size: int,
        pool_atom_feats: bool = True,
        pool_bond_feats: bool = True,
        pool_global_feats: bool = True,
        reducer="mean",
        num_iterations=6,
        num_layers=3,
    ):
        self.num_iterations = num_iterations
        self.num_layers = num_layers
        super().__init__(
            in_size, pool_atom_feats, pool_bond_feats, pool_global_feats, reducer
        )

    def init_atom_pool_method(self):
        method = Set2Set(
            input_dim=self.in_size,
            n_iters=self.num_iterations,
            n_layers=self.num_layers,
        )
        out_size = 2 * self.in_size

        return method, out_size

    def init_bond_pool_method(self):
        return self.init_atom_pool_method()


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
        Compute set2set pool.

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


class AttentiveReduce(nn.Module):
    """
    A pooling layer similar to the GAT:

    parameter vector w:

    hi = leakyrelu(hi*w) or hi = sigmoid(hi*w)
    alpha_i = softmax(hi)
    readout = sum_i alpha_i * hi

    in which hi is the feature of atom/bond i.

    This is also very similar to dgl.nn.WeightAndSum, where sigmoid is used, but here
    we use LeakyRelu.

    We do not use sigmoid because for large molecules there may only be a few atoms
    that matter. sigmoid forces a score range of 0~1 which may reduce the importance of
    the atom that really matter when passed through the softmax.

    Args:
        reducer: reduce method. If `sum`, do exactly the above. If `mean`, the readout
            is obtained as the weighted mean, not weight sum.
    """

    def __init__(self, in_size, activation: str = "LeakyReRU", reducer="sum"):
        super().__init__()
        assert reducer in [
            "sum",
            "mean",
        ], f"Expect reducer be sum or mean; got {reducer}"
        self.reducer = reducer

        if activation == "LeakyReLU":
            act = nn.LeakyReLU(negative_slope=0.2)
        elif activation == "Sigmoid":
            act = nn.Sigmoid()
        else:
            raise ValueError(
                "Expect activation for AttentiveReduce is `LeakyReLU` or `Sigmoid`;"
                f"got {activation}"
            )

        self.mlp = nn.Sequential(nn.Linear(in_size, 1, bias=False), act)

    def forward(self, feat: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: feature tensor of shape (N, D) where N is the total number of features,
                and D is the feature dimension.
            sizes: 1D tensor (shape (B,)) of the size of the features for each graph.
                sum(sizes) should be equal to D.
        """
        alpha = self.get_attention_score(feat, sizes)
        out = segment_reduce(sizes, feat * alpha, reducer=self.reducer)

        return out

    def get_attention_score(self, feat: torch.Tensor, sizes: torch.Tensor):
        """
        Returns:
            2D tensor of shape (N, 1), each is an attention score for an atom/bond.
        """
        feat = self.mlp(feat)
        alpha = segment_softmax(sizes, feat)

        return alpha


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


def get_reaction_feature_pooling(
    pool_method,
    in_size: int,
    pool_atom_feats: bool = True,
    pool_bond_feats: bool = True,
    pool_global_feats: bool = True,
    pool_kwargs: Dict[str, Any] = None,
) -> BasePooling:
    """

    Args:
        pool_method:
        in_size:
        pool_atom_feats:
        pool_bond_feats:
        pool_global_feats:
        pool_kwargs: extra kwargs for pooling method. e.g. Set2Set and AttentiveReduce.

    Returns:
        A callable to return pooled reaction features
    """

    if pool_method in ["reduce_sum", "reduce_mean"]:
        reducer = pool_method.split("_")[-1]
        return ReducePool(
            in_size, pool_atom_feats, pool_bond_feats, pool_global_feats, reducer
        )

    elif pool_method in ["center_reduce_sum", "center_reduce_mean"]:
        reducer = pool_method.split("_")[-1]
        return CenterReducePool(
            in_size, pool_atom_feats, pool_bond_feats, pool_global_feats, reducer
        )

    elif pool_method in ["attentive_reduce_sum", "attentive_reduce_mean"]:
        reducer = pool_method.split("_")[-1]

        if pool_kwargs is None:
            activation = "LeakyReLU"
        else:
            activation = pool_kwargs["activation"]

        return AttentiveReducePool(
            in_size,
            pool_atom_feats,
            pool_bond_feats,
            pool_global_feats,
            reducer=reducer,
            activation=activation,
        )

    elif pool_method == "set2set":
        if pool_kwargs is None:
            num_iterations = 6
            num_layers = 3
        else:
            num_iterations = pool_kwargs["set2set_num_iterations"]
            num_layers = pool_kwargs["set2set_num_layers"]

        return Set2SetPool(
            in_size,
            pool_atom_feats,
            pool_bond_feats,
            pool_global_feats,
            num_iterations=num_iterations,
            num_layers=num_layers,
        )

    else:
        raise ValueError(f"Unaccepted pooling method {pool_method}")
