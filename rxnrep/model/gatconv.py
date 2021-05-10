"""
GatedConv as in gatedconv.py (where bond is represented as node), but here bonds are
represented as edges.
"""

from typing import Callable, Dict, Union

import dgl
import torch
from dgl import function as fn
from torch import nn

from rxnrep.model.utils import MLP, get_activation


class GATConv(nn.Module):
    """
    GAT conv with bond feats.

    Bond feats are added to atom feats of src node, and do not do explicit bond feats
    update.
    This is a reimplementation of https://github.com/snap-stanford/pretrain-gnns/blob/7bb81b5cc2d37241ee72cbfa40fbd89b0cc2394f/chem/model.py#L107
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        num_fc_layers: int = 1,
        batch_norm: bool = True,
        activation: Callable = nn.ReLU(),
        out_batch_norm: bool = True,
        out_activation: Callable = nn.ReLU(),
        residual: bool = False,
        dropout: Union[float, None] = None,
        num_heads: int = 2,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.out_size = out_size

        # fc layer
        hidden_size = [out_size] * (num_fc_layers - 1)
        self.fc_layer = MLP(
            in_size=in_size,
            hidden_sizes=hidden_size,
            batch_norm=batch_norm,
            activation=activation,
            out_size=out_size * num_heads,
            out_bias=False,
        )

        # parameters for attention
        self.attn_l = nn.Parameter(torch.zeros(1, num_heads, out_size))
        self.attn_r = nn.Parameter(torch.zeros(1, num_heads, out_size))
        self.reset_parameters()
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        if out_batch_norm:
            self.out_batch_norm = True
            self.bn_node_h = nn.BatchNorm1d(out_size)
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
            g: the graphm bond as edges
            feats: {name, feats}. Atom, bond, and global features.

        Returns:
            updated features.
        """

        g = g.local_var()

        h = feats["atom"]
        e = feats["bond"]

        #
        # update features
        #
        g.nodes["atom"].data.update({"h": h})
        g.edges["bond"].data.update({"e": e})

        g.apply_edges(fn.u_add_e("h", "e", "e"), etype="bond")
        h_add_e = g.edges["bond"].data["e"]
        h_add_e = self.fc_layer(h_add_e).view(
            -1, self.num_heads, self.out_size
        )  # (N, heads, outsize)
        h = self.fc_layer(h).view(-1, self.num_heads, self.out_size)

        el = (h_add_e * self.attn_l).sum(dim=-1).unsqueeze(-1)  # (N, heads, 1)
        er = (h * self.attn_r).sum(dim=-1).unsqueeze(-1)  # (N, heads, 1)

        g.edges["bond"].data.update({"ft": h_add_e, "el": el})
        g.nodes["atom"].data.update({"er": er})

        # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
        g.apply_edges(fn.e_add_v("el", "er", "e"), etype="bond")
        e = self.leaky_relu(g.edges["bond"].data.pop("e"))  # (N, heads, 1)

        # compute softmax
        alpha = edge_softmax(g, e)  # (N, heads, 1)
        g.edges["bond"].data["alpha"] = alpha

        # message passing
        g.update_all(
            lambda edges: {"m": edges.data["ft"] * edges.data["alpha"]},
            fn.sum("m", "ft"),
            etype="bond",
        )
        h = g.nodes["atom"].data["ft"]  # (N, heads, out_size)

        # mean of heads
        h = torch.mean(h, dim=1)  # (N, out_size)

        if self.out_batch_norm:
            h = self.bn_node_h(h)
        if self.out_activation:
            h = self.out_activation(h)
        if self.residual:
            h = feats["atom"] + h

        # dropout
        if self.dropout:
            h = self.dropout(h)

        feats = {"atom": h, "bond": feats["bond"]}

        return feats

    def reset_parameters(self):
        """Reinitialize parameters."""
        gain = nn.init.calculate_gain("relu")
        # for nt, layer in self.fc_layers.items():
        #     nn.init.xavier_normal_(layer.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)


def edge_softmax(graph, edata, etype="bond"):
    """
    Edge softmax for a specific edge type.

    Args:
        graph:
        edata:
        etype:

    Returns:
    """
    g = graph.local_var()

    g.edges[etype].data["e"] = edata

    ## The softmax trick, making the exponential stable.
    ## see https://stackoverflow.com/questions/42599498/numercially-stable-softmax
    g.update_all(fn.copy_e("e", "m"), fn.max("m", "emax"), etype=etype)
    # subtract max and compute exponential
    g.apply_edges(fn.e_sub_v("e", "emax", "e"), etype=etype)
    g.edges[etype].data["out"] = torch.exp(g.edges[etype].data["e"])

    # e sum
    g.update_all(fn.copy_e("out", "m"), fn.sum("m", "out_sum"), etype=etype)

    g.apply_edges(fn.e_div_v("out", "out_sum", "a"), etype=etype)
    alpha = g.edges[etype].data["a"]

    return alpha
