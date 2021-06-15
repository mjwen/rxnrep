"""
GAT conv with bond and global features.
"""

from typing import Callable, Dict, Union

import dgl
import torch
from dgl import function as fn
from torch import nn

from rxnrep.layer.utils import MLP, get_activation


class GATConvGlobal(nn.Module):
    """
    GAT conv with bond and global feature update.

    Bond update the same as GatedConv, i.e. no attention, but sum connected atom atom
    features, global features and the bond feature itself. The reason for this each
    bond will have two atoms, both of which should be important.

    Atom and global feature update uses attention.
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
        self.bond_layer = _GATConvGlobalBondLayer(
            in_size=in_size,
            out_size=out_size,
            num_fc_layers=num_fc_layers,
            batch_norm=batch_norm,
            activation=activation,
            out_batch_norm=out_batch_norm,
            out_activation=out_activation,
            residual=residual,
        )

        self.atom_layer = _GATConvGlobalAtomAttention(
            in_size=in_size,
            out_size=out_size,
            num_fc_layers=num_fc_layers,
            batch_norm=batch_norm,
            activation=activation,
            out_batch_norm=out_batch_norm,
            out_activation=out_activation,
            residual=residual,
            num_heads=num_heads,
            negative_slope=negative_slope,
        )

        self.global_layer = _GATConvGlobalGlobalAttention(
            in_size=in_size,
            out_size=out_size,
            num_fc_layers=num_fc_layers,
            batch_norm=batch_norm,
            activation=activation,
            out_batch_norm=out_batch_norm,
            out_activation=out_activation,
            residual=residual,
            num_heads=num_heads,
            negative_slope=negative_slope,
        )

        delta = 1e-2
        if dropout and dropout > delta:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(
        self,
        g: dgl.DGLGraph,
        feats: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            g: bond as edge graph
            feats: {name, feats}. Atom, bond, and global features.

        Returns:
            updated features.
        """

        updated_feats = {k: v for k, v in feats.items()}

        e = self.bond_layer(g, updated_feats)
        updated_feats["bond"] = e

        h = self.atom_layer(g, updated_feats)
        updated_feats["atom"] = h

        u = self.global_layer(g, updated_feats)

        # dropout
        if self.dropout:
            h = self.dropout(h)
            e = self.dropout(e)
            u = self.dropout(u)

        return {"atom": h, "bond": e, "global": u}


class _GATConvGlobalBondLayer(nn.Module):
    """
    Bond layer GATConvGlobal.

    Bond update the same as GatedConv, i.e. no attention, but sum connected atom atom
    features, global features and the bond feature itself. The reason for this each
    bond will have two atoms, both of which should be important.
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
    ):
        super().__init__()

        # fc layer
        hidden_size = [out_size] * (num_fc_layers - 1)
        self.mlp_bond = MLP(
            in_size=in_size,
            hidden_sizes=hidden_size,
            batch_norm=batch_norm,
            activation=activation,
            out_size=out_size,
            out_bias=False,
        )

        self.mlp_atom = MLP(
            in_size=in_size,
            hidden_sizes=hidden_size,
            batch_norm=batch_norm,
            activation=activation,
            out_size=out_size,
            out_bias=False,
        )

        self.mlp_global = MLP(
            in_size=in_size,
            hidden_sizes=hidden_size,
            batch_norm=batch_norm,
            activation=activation,
            out_size=out_size,
            out_bias=False,
        )

        if out_batch_norm:
            self.out_batch_norm = True
            self.bn_node_e = nn.BatchNorm1d(out_size)
        else:
            self.out_batch_norm = False

        if out_activation:
            self.out_activation = get_activation(out_activation)
        else:
            self.out_activation = None

        self.residual = residual
        if in_size != out_size:
            self.residual = False

    def forward(
        self,
        g: dgl.DGLGraph,
        feats: Dict[str, torch.Tensor],
    ) -> torch.Tensor:

        g = g.local_var()

        h = self.mlp_atom(feats["atom"])  # (Na, out_size)
        e = self.mlp_bond(feats["bond"])  # (Nb, out_size)
        u = self.mlp_global(feats["global"])  # (Ng, out_size)

        #
        # update bond feature
        #
        g.nodes["atom"].data["h"] = h
        g.nodes["global"].data["u"] = u

        # global feats to edge
        # step 1, a simpy copy to atom node (sum operates on 1 tensor since there is
        # only 1 global node connected to each atom node)
        # step 2, copy from atom node to edge, only need to grab from src
        g.update_all(fn.copy_u("u", "m"), fn.sum("m", "u"), etype="g2a")
        g.apply_edges(fn.copy_u("u", "u"), etype="bond")
        u_at_e = g.edges["bond"].data.pop("u")

        # sum of atom feats to edge
        g.apply_edges(fn.u_add_v("h", "h", "sum_h"), etype="bond")
        sum_h = g.edges["bond"].data.pop("sum_h")

        # aggregate
        e = sum_h + e + u_at_e

        if self.out_batch_norm:
            e = self.bn_node_e(e)
        if self.out_activation:
            e = self.out_activation(e)
        if self.residual:
            e = feats["bond"] + e

        return e


class _GATConvGlobalAtomAttention(nn.Module):
    """
    Atom attention part of GATConvGlobal.
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
        num_heads: int = 2,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.out_size = out_size

        # fc layers
        hidden_size = [out_size] * (num_fc_layers - 1)
        self.mlp_bond = MLP(
            in_size=in_size,
            hidden_sizes=hidden_size,
            batch_norm=batch_norm,
            activation=activation,
            out_size=out_size * num_heads,
            out_bias=False,
        )

        self.mlp_atom = MLP(
            in_size=in_size,
            hidden_sizes=hidden_size,
            batch_norm=batch_norm,
            activation=activation,
            out_size=out_size * num_heads,
            out_bias=False,
        )

        self.mlp_global = MLP(
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
            self.out_activation = None

        self.residual = residual
        if in_size != out_size:
            self.residual = False

    def forward(
        self,
        g: dgl.DGLGraph,
        feats: Dict[str, torch.Tensor],
    ) -> torch.Tensor:

        g = g.local_var()

        h = self.mlp_atom(feats["atom"]).view(
            -1, self.num_heads, self.out_size
        )  # (Na, H, out_size)
        e = self.mlp_bond(feats["bond"]).view(
            -1, self.num_heads, self.out_size
        )  # (Nb, H, out_size)
        u = self.mlp_global(feats["global"]).view(
            -1, self.num_heads, self.out_size
        )  # (Ng, H, out_size)

        # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
        # master feat
        er = (h * self.attn_r).sum(dim=-1).unsqueeze(-1)  # (Na, heads, 1)
        el_h = (h * self.attn_l).sum(dim=-1).unsqueeze(-1)  # (Na, heads, 1)
        el_e = (e * self.attn_l).sum(dim=-1).unsqueeze(-1)  # (Nb, heads, 1)
        el_u = (u * self.attn_l).sum(dim=-1).unsqueeze(-1)  # (Ng, heads, 1)

        g.nodes["atom"].data.update({"ft": h, "el": el_h, "er": er})
        g.edges["bond"].data.update({"ft": e, "el": el_e})
        g.nodes["global"].data.update({"ft": u, "el": el_u})

        # compute leaky attention

        g.apply_edges(fn.u_add_v("el", "er", "eh"), etype="bond")
        eh = self.leaky_relu(g.edges["bond"].data.pop("eh"))  # (Nb, heads, 1)
        g.edges["bond"].data["eh"] = eh

        g.apply_edges(fn.e_add_v("el", "er", "ee"), etype="bond")
        ee = self.leaky_relu(g.edges["bond"].data.pop("ee"))  # (Nb, heads, 1)
        g.edges["bond"].data["ee"] = ee

        g.apply_edges(fn.u_add_v("el", "er", "eu"), etype="g2a")
        eu = self.leaky_relu(g.edges["g2a"].data.pop("eu"))  # (Ng, heads, 1)
        g.edges["g2a"].data["eu"] = eu

        #
        # step 1
        # compute neighboring atom, bonds,m and global attention score

        # The softmax trick, making the exponential stable.
        # see https://stackoverflow.com/questions/42599498/numercially-stable-softmax
        g.edges["bond"].data["eh_ee_max"] = torch.maximum(eh, ee)
        g.multi_update_all(
            {
                "bond": (fn.copy_e("eh_ee_max", "m"), fn.max("m", "emax")),  # eh and ee
                "g2a": (fn.copy_e("eu", "m"), fn.max("m", "emax")),  # global
            },
            "max",
        )
        # subtract and compute exponential
        g.apply_edges(fn.e_sub_v("eh", "emax", "eh"), etype="bond")
        g.apply_edges(fn.e_sub_v("ee", "emax", "ee"), etype="bond")
        g.apply_edges(fn.e_sub_v("eu", "emax", "eu"), etype="g2a")
        g.edges["bond"].data["eh"] = torch.exp(g.edges["bond"].data["eh"])
        g.edges["bond"].data["ee"] = torch.exp(g.edges["bond"].data["ee"])
        g.edges["g2a"].data["eu"] = torch.exp(g.edges["g2a"].data["eu"])

        # e sum
        eh_ee_sum = g.edges["bond"].data["eh"] + g.edges["bond"].data["ee"]
        g.edges["bond"].data["eh_ee_sum"] = eh_ee_sum
        g.multi_update_all(
            {
                "bond": (fn.copy_e("eh_ee_sum", "m"), fn.sum("m", "esum")),  # eh and ee
                "g2a": (fn.copy_e("eu", "m"), fn.sum("m", "esum")),  # global
            },
            "sum",
        )
        # attention score
        g.apply_edges(fn.e_div_v("eh", "esum", "alpha_h"), etype="bond")
        g.apply_edges(fn.e_div_v("ee", "esum", "alpha_e"), etype="bond")
        g.apply_edges(fn.e_div_v("eu", "esum", "alpha_u"), etype="g2a")

        # step 2
        # aggregate features

        # message passing, "ft" is of shape(H, out), and "a" is of shape(H, 1)
        # computing the part inside the parenthesis of eq. 4 of the GAT paper
        g.apply_edges(fn.u_mul_e("ft", "alpha_h", "m"), etype="bond")
        ft_alpha_h = g.edges["bond"].data["m"]
        ft_alpha_e = g.edges["bond"].data["ft"] * g.edges["bond"].data["alpha_e"]
        g.edges["bond"].data["ft_alpha_h_alpha_e_sum"] = ft_alpha_h + ft_alpha_e

        g.multi_update_all(
            {
                "bond": (fn.copy_e("ft_alpha_h_alpha_e_sum", "m"), fn.sum("m", "out")),
                "g2a": (fn.u_mul_e("ft", "alpha_u", "m"), fn.sum("m", "out")),
            },
            "sum",
        )

        h = g.nodes["atom"].data["out"]  # (N, heads, out_size)

        # mean of heads
        h = torch.mean(h, dim=1)  # (N, out_size)

        if self.out_batch_norm:
            h = self.bn_node_h(h)
        if self.out_activation:
            h = self.out_activation(h)
        if self.residual:
            h = feats["atom"] + h

        return h

    def reset_parameters(self):
        """Reinitialize parameters."""
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)


class _GATConvGlobalGlobalAttention(nn.Module):
    """
    Global feats attention part of GATConvGlobal.

    This is really cumbersome since info has to travel long distance from bond edge to
    global node and vice versa, which is in turn due to the fact bond is modelled as as
    edge graph.
    Much easier to do if we use bond as node graph. Should switch to it at some point.
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
        num_heads: int = 2,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.out_size = out_size

        # fc layers
        hidden_size = [out_size] * (num_fc_layers - 1)
        self.mlp_bond = MLP(
            in_size=in_size,
            hidden_sizes=hidden_size,
            batch_norm=batch_norm,
            activation=activation,
            out_size=out_size * num_heads,
            out_bias=False,
        )

        self.mlp_atom = MLP(
            in_size=in_size,
            hidden_sizes=hidden_size,
            batch_norm=batch_norm,
            activation=activation,
            out_size=out_size * num_heads,
            out_bias=False,
        )

        self.mlp_global = MLP(
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
            self.bn_node_u = nn.BatchNorm1d(out_size)
        else:
            self.out_batch_norm = False

        if out_activation:
            self.out_activation = get_activation(out_activation)
        else:
            self.out_activation = None

        self.residual = residual
        if in_size != out_size:
            self.residual = False

    def forward(
        self,
        g: dgl.DGLGraph,
        feats: Dict[str, torch.Tensor],
    ) -> torch.Tensor:

        g = g.local_var()

        h = self.mlp_atom(feats["atom"]).view(
            -1, self.num_heads, self.out_size
        )  # (Na, H, out_size)
        e = self.mlp_bond(feats["bond"]).view(
            -1, self.num_heads, self.out_size
        )  # (Nb, H, out_size)
        u = self.mlp_global(feats["global"]).view(
            -1, self.num_heads, self.out_size
        )  # (Ng, H, out_size)

        # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
        er = (u * self.attn_r).sum(dim=-1).unsqueeze(-1)  # (Nu, heads, 1)
        el_h = (h * self.attn_l).sum(dim=-1).unsqueeze(-1)  # (Na, heads, 1)
        el_e = (e * self.attn_l).sum(dim=-1).unsqueeze(-1)  # (Nb, heads, 1)
        el_u = (u * self.attn_l).sum(dim=-1).unsqueeze(-1)  # (Nu, heads, 1)

        g.nodes["global"].data.update({"ft": u, "er": er})
        g.nodes["atom"].data.update({"ft": h})
        g.edges["bond"].data.update({"ft": e, "el_e": el_e})

        # compute leaky attention

        g.update_all(fn.copy_u("er", "m"), fn.sum("m", "er"), etype="g2a")  # copy er

        # el + er for e (info stored on bond edge)
        g.apply_edges(fn.e_add_v("el_e", "er", "ee"), etype="bond")
        ee = self.leaky_relu(g.edges["bond"].data.pop("ee"))  # (Nb, heads, 1)
        g.edges["bond"].data["ee"] = ee

        # el+er for h (info stored on atom node)
        eh = el_h + g.nodes["atom"].data["er"]
        eh = self.leaky_relu(eh)  # (Na, heads, 1)

        # el+er for u (info explicitly stored)
        eu = el_u + er
        eu = self.leaky_relu(eu)  # (Nu, heads, 1)

        #
        # step 1
        # compute neighboring atom, bonds,m and global attention score

        # The softmax trick, making the exponential stable.
        # see https://stackoverflow.com/questions/42599498/numercially-stable-softmax

        # trick step 1 find max
        # ee max
        g.update_all(fn.copy_e("ee", "m"), fn.max("m", "ee_max"), etype="bond")

        # ee and eh max on atom node
        ee_max = g.nodes["atom"].data["ee_max"]
        g.nodes["atom"].data["ee_eh_max"] = torch.maximum(ee_max, eh)

        # ee eh, and eu max on global node
        g.update_all(fn.copy_u("ee_eh_max", "m"), fn.max("m", "ee_eh_max"), etype="a2g")
        ee_eh_max = g.nodes["global"].data["ee_eh_max"]
        ee_eh_eu_max = torch.maximum(ee_eh_max, eu)
        g.nodes["global"].data["emax"] = ee_eh_eu_max

        # trick step2, subtract emax and computed exponential
        g.update_all(
            fn.copy_u("emax", "m"), fn.sum("m", "emax"), etype="g2a"
        )  # copy e max

        # subtract and compute exponential
        g.apply_edges(fn.e_sub_v("ee", "emax", "ee"), etype="bond")
        eh = eh - g.nodes["atom"].data["emax"]
        eu = eu - g.nodes["global"].data["emax"]

        g.edges["bond"].data["ee"] = torch.exp(g.edges["bond"].data["ee"])
        eh = torch.exp(eh)
        eu = torch.exp(eu)

        # e sum

        # ee sum
        g.update_all(fn.copy_e("ee", "m"), fn.sum("m", "ee_sum"), etype="bond")

        #  ee eh sum at node (0.5 because each bond is represented by 2 edges)
        ee_sum = 0.5 * g.nodes["atom"].data["ee_sum"]
        g.nodes["atom"].data["ee_eh_sum"] = ee_sum + eh

        # ee eh eu sum
        g.update_all(fn.copy_u("ee_eh_sum", "m"), fn.sum("m", "ee_eh_sum"), etype="a2g")
        ee_eh_eu_sum = g.nodes["global"].data["ee_eh_sum"] + eu
        g.nodes["global"].data["esum"] = ee_eh_eu_sum

        # compute attention score

        g.update_all(
            fn.copy_u("esum", "m"), fn.sum("m", "esum"), etype="g2a"
        )  # copy e sum

        g.apply_edges(fn.e_div_v("ee", "esum", "alpha_e"), etype="bond")
        alpha_h = eh / g.nodes["atom"].data["esum"]
        alpha_u = eu / g.nodes["global"].data["esum"]

        # ft alpha e
        g.edges["bond"].data["ft_alpha_e"] = (
            g.edges["bond"].data["ft"] * g.edges["bond"].data["alpha_e"]
        )

        g.update_all(
            fn.copy_e("ft_alpha_e", "m"), fn.sum("m", "ft_alpha_e_sum"), etype="bond"
        )

        # ft alpha e h sum at atom node (0.5 because each bond is represented by 2 edges)
        ft_alpha_h = alpha_h * g.nodes["atom"].data["ft"]
        g.nodes["atom"].data["ft_alpha_e_h_sum"] = (
            0.5 * g.nodes["atom"].data["ft_alpha_e_sum"] + ft_alpha_h
        )

        # ft alpha e, h, u sum
        g.update_all(
            fn.copy_u("ft_alpha_e_h_sum", "m"),
            fn.sum("m", "ft_alpha_e_h_sum"),
            etype="a2g",
        )

        ft_alpha_u = alpha_u * g.nodes["global"].data["ft"]
        ft_alpha_e_h_u_sum = g.nodes["global"].data["ft_alpha_e_h_sum"] + ft_alpha_u

        u = ft_alpha_e_h_u_sum
        u = torch.mean(u, dim=1)  # (N, out_size)

        if self.out_batch_norm:
            u = self.bn_node_u(u)
        if self.out_activation:
            u = self.out_activation(u)
        if self.residual:
            u = feats["global"] + u

        return u

    def reset_parameters(self):
        """Reinitialize parameters."""
        gain = nn.init.calculate_gain("relu")
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
