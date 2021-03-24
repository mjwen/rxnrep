"""
GIN as in Strategies for pretraining graph neural networks.
This implements the for protein function prediction model.
"""

from typing import Dict, Union

import dgl
import torch
from dgl import function as fn
from torch import nn

from rxnrep.model.utils import MLP, get_activation


class GINConv(nn.Module):
    """
    The protein function prediction model as in:
    Strategies for pretraining graph neural networks.

    Note, we do not use self-loop edge as in the paper, because it cannot be easily
    such edges cannot be easily featurized. We do use self-loop atom nodes.

    Args:
        out_batch_norm: batch norm for output
        out_activation: activation for output
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        num_fc_layers: int = 2,
        batch_norm: bool = True,
        activation: str = "ReLU",
        out_batch_norm: bool = False,
        out_activation: str = None,
        residual: bool = False,
        dropout: Union[float, None] = None,
    ):
        super().__init__()

        hidden_sizes = [out_size * 2] * (num_fc_layers - 1)

        self.mlp_bond = MLP(
            2 * in_size,
            hidden_sizes,
            batch_norm=batch_norm,
            activation=activation,
            out_size=out_size,
        )
        self.mlp_atom = MLP(
            in_size + out_size,  # in_size: atom; out_size: bond
            hidden_sizes,
            batch_norm=batch_norm,
            activation=activation,
            out_size=out_size,
        )

        if out_batch_norm:
            self.out_batch_norm = True
            self.bn_atom = nn.BatchNorm1d(out_size)
            self.bn_bond = nn.BatchNorm1d(out_size)
        else:
            self.out_batch_norm = False

        if out_activation:
            self.out_activation = get_activation(out_activation)
        else:
            self.out_activation = False

        self.residual = residual
        if in_size != out_size:
            self.residual = False

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
            g: the bond as edges graph
            feats: {name, feats}. Atom, bond, and global features.

        Returns:
            updated features.
        """

        g = g.local_var()

        h = feats["atom"]
        e = feats["bond"]

        #
        # update bond features
        #
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

        #
        # update atom features
        #
        g.edges["bond"].data.update({"e": e})

        g.update_all(fn.copy_u("h", "m"), fn.sum("m", "sum_h"), etype="bond")
        g.update_all(fn.copy_e("e", "m"), fn.sum("m", "sum_e"), etype="bond")
        sum_h = g.nodes["atom"].data.pop("sum_h") + h  # + h for self loop
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


class GINConvGlobal(nn.Module):
    """
    The protein function prediction model as in:
    Strategies for pretraining graph neural networks.

    Besides atom and bond features, we add the support of global features.

    Args:
        out_batch_norm: batch norm for output
        out_activation: activation for output
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        num_fc_layers: int = 2,
        batch_norm: bool = True,
        activation: str = "ReLU",
        out_batch_norm: bool = False,
        out_activation: str = None,
        residual: bool = False,
        dropout: Union[float, None] = None,
    ):
        super().__init__()

        hidden_sizes = [out_size * 3] * (num_fc_layers - 1)
        self.mlp_bond = MLP(
            3 * in_size,
            hidden_sizes,
            batch_norm=batch_norm,
            activation=activation,
            out_size=out_size,
        )

        self.mlp_atom = MLP(
            2 * in_size + out_size,  # in_size: atom, global; out_size: bond
            hidden_sizes,
            batch_norm=batch_norm,
            activation=activation,
            out_size=out_size,
        )

        self.mlp_global = MLP(
            in_size + 2 * out_size,  # in_size: global; out_size: bond, atom
            hidden_sizes,
            batch_norm=batch_norm,
            activation=activation,
            out_size=out_size,
        )

        if out_batch_norm:
            self.out_batch_norm = True
            self.bn_atom = nn.BatchNorm1d(out_size)
            self.bn_bond = nn.BatchNorm1d(out_size)
        else:
            self.out_batch_norm = False

        if out_activation:
            self.out_activation = get_activation(out_activation)
        else:
            self.out_activation = False

        self.residual = residual
        if in_size != out_size:
            self.residual = False

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
            g: the bond as edges graph
            feats: {name, feats}. Atom, bond, and global features.

        Returns:
            updated features.
        """

        g = g.local_var()

        h = feats["atom"]
        e = feats["bond"]
        u = feats["global"]

        #
        # update bond features
        #
        g.nodes["atom"].data.update({"h": h})
        g.edges["bond"].data.update({"e": e})
        g.nodes["global"].data.update({"u": u})

        # global feats to edge
        # step 1, a simpy copy to atom node (sum operates on 1 tensor since there is
        # only 1 global node connected to each atom node)
        # step 2, copy from atom node to edge, only need to grab from src
        g.update_all(fn.copy_u("u", "m"), fn.sum("m", "u_node"), etype="g2a")
        g.apply_edges(fn.copy_u("u_node", "u_edge"), etype="bond")
        u_edge = g.edges["bond"].data.pop("u_edge")

        # sum of atom feats to edge
        g.apply_edges(fn.u_add_v("h", "h", "sum_h"), etype="bond")
        sum_h = g.edges["bond"].data.pop("sum_h")

        # aggregate feats
        e = self.mlp_bond(torch.cat((sum_h, e, u_edge), dim=-1))

        if self.out_batch_norm:
            e = self.bn_bond(e)
        if self.out_activation:
            e = self.out_activation(e)
        if self.residual:
            e = feats["bond"] + e
        if self.dropout:
            e = self.dropout(e)

        #
        # update atom features
        #
        g.edges["bond"].data.update({"e": e})

        # sum updated edge feats to atom node
        g.update_all(fn.copy_e("e", "m"), fn.sum("m", "sum_e"), etype="bond")
        sum_e = g.nodes["atom"].data["sum_e"]

        # sum neighboring atom node feats to atom node
        g.update_all(fn.copy_u("h", "m"), fn.sum("m", "sum_h"), etype="bond")
        sum_h = g.nodes["atom"].data.pop("sum_h") + h  # + h for self loop

        # global feats (u_node already stored in atom node)
        u_node = g.nodes["atom"].data.pop("u_node")

        # aggregate
        h = self.mlp_atom(torch.cat((sum_h, sum_e, u_node), dim=-1))

        if self.out_batch_norm:
            h = self.bn_atom(h)
        if self.out_activation:
            h = self.out_activation(h)
        if self.residual:
            h = feats["atom"] + h
        if self.dropout:
            h = self.dropout(h)

        #
        # update global feats
        #
        g.nodes["atom"].data.update({"h": h})

        # edge features to global (e_sum already stored in atom node)
        g.update_all(fn.copy_u("sum_e", "m"), fn.sum("m", "sum_e"), etype="a2g")
        sum_e = g.nodes["global"].data.pop("sum_e")
        sum_e = 0.5 * sum_e  # 0.5 * because each bond is represented by two edges

        # atom node features to global
        g.update_all(fn.copy_u("h", "m"), fn.sum("m", "sum_h"), etype="a2g")
        sum_h = g.nodes["global"].data.pop("sum_h")

        # aggregate
        u = self.mlp_global(torch.cat((sum_h, sum_e, u), dim=-1))

        feats = {"atom": h, "bond": e, "global": u}

        return feats
