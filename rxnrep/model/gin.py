"""
GIN as in Strategies for pretraining graph neural networks.
This implements the for protein function prediction model.
"""

from typing import Callable, Dict, Union

import dgl
import torch
from dgl import function as fn
from torch import nn

from rxnrep.model.utils import MLP, get_activation


class GINConv(nn.Module):
    """
    The protein function prediction model as in:
    Strategies for pretraining graph neural networks.

    Args:
        out_batch_norm: batch norm for output
        out_activation: activation for output
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_fc_layers: int = 2,
        batch_norm: bool = True,
        activation: str = "ReLU",
        out_batch_norm: bool = False,
        out_activation: str = None,
        residual: bool = False,
        dropout: Union[float, None] = None,
    ):
        super().__init__()

        hidden_sizes = [output_dim * 2] * (num_fc_layers - 1)
        self.mlp_atom = MLP(
            2 * input_dim,
            hidden_sizes,
            batch_norm=batch_norm,
            activation=activation,
            out_size=output_dim,
        )
        self.mlp_bond = MLP(
            2 * input_dim,
            hidden_sizes,
            batch_norm=batch_norm,
            activation=activation,
            out_size=output_dim,
        )

        if out_batch_norm:
            self.out_batch_norm = True
            self.bn_atom = nn.BatchNorm1d(output_dim)
            self.bn_bond = nn.BatchNorm1d(output_dim)
        else:
            self.out_batch_norm = False

        if out_activation:
            self.out_activation = get_activation(out_activation)
        else:
            self.out_activation = False

        self.residual = residual

        delta = 1e-2
        if dropout and dropout > delta:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = False

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

        # update bond features
        g.nodes["atom"].data.update({"h": h})
        g.edges["bond"].data.update({"e": e})

        g.apply_edges(fn.u_add_v("h", "h", "sum_h"), etype="bond")
        sum_h = g.edges["bond"].data.pop("sum_h")
        e = self.mlp_bond(torch.cat((sum_h, e), dim=-1))

        if self.out_batch_norm:
            e = self.bn_bond(e)
        if self.out_activation:
            e = self.out_activation(e)
        if self.residual:
            e = feats["bond"] + e
        if self.dropout:
            e = self.dropout(e)

        # update atom features
        g.edges["bond"].data.update({"e": e})

        g.update_all(fn.copy_u("h", "m"), fn.sum("m", "sum_h"), etype="bond")
        g.update_all(fn.copy_e("e", "m"), fn.sum("m", "sum_e"), etype="bond")
        sum_h = g.nodes["atom"].data.pop("sum_h")
        sum_e = g.nodes["atom"].data.pop("sum_e")
        h = self.mlp_atom(torch.cat((sum_h, sum_e), dim=-1))

        if self.out_batch_norm:
            h = self.bn_atom(h)
        if self.out_activation:
            h = self.out_activation(h)
        if self.residual:
            h = feats["atom"] + h
        if self.dropout:
            h = self.dropout(h)

        feats = {"atom": h, "bond": e}

        return feats


class GINConvWithGlobal(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_hidden_layers: int = 2,
        batch_norm: bool = True,
        activation: Callable = nn.ReLU(),
        residual: bool = False,
        dropout: Union[float, None] = None,
        has_global_feature: bool = False,
    ):
        self.has_global_feature = has_global_feature
        self.batch_norm = batch_norm
        self.residual = residual

        hidden_sizes = [output_dim] * num_hidden_layers
        self.mlp_atom = MLP(input_dim, hidden_sizes, activation=activation)
        self.mlp_bond = MLP(input_dim, hidden_sizes, activation=activation)
        if has_global_feature:
            self.mlp_bond = MLP(input_dim, hidden_sizes, activation=activation)

        if self.batch_norm:
            self.bn_atom = nn.BatchNorm1d(output_dim)
            self.bn_bond = nn.BatchNorm1d(output_dim)
            if has_global_feature:
                self.bn_global = nn.BatchNorm1d(output_dim)

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
        if self.has_global_feature:
            u = feats["global"]

        # for residual connection
        h_in = h
        e_in = e
        u_in = u

        # update bond features
