"""
Readout (pool) layers.
"""
from typing import Any, Callable, Dict, List, Tuple

import dgl
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
        if reducer not in ["sum", "mean", None]:
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
        reducer: str = "sum",
        activation: str = "Sigmoid",
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

    def __init__(self, in_size, activation: str = "Sigmoid", reducer="sum"):
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


def get_reaction_feature_pooling(
    pool_method,
    in_size: int,
    pool_atom_feats: bool = True,
    pool_bond_feats: bool = True,
    pool_global_feats: bool = True,
) -> BasePooling:
    """

    Args:
        pool_method:
        in_size:
        pool_atom_feats:
        pool_bond_feats:
        pool_global_feats:

    Returns:
        A callable to return pooled reaction features
    """

    if pool_method in ["attentive_reduce_sum", "attentive_reduce_mean"]:
        reducer = pool_method.split("_")[-1]

        return AttentiveReducePool(
            in_size,
            pool_atom_feats,
            pool_bond_feats,
            pool_global_feats,
            reducer=reducer,
        )

    else:
        raise ValueError(f"Unaccepted pooling method {pool_method}")
