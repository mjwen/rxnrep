import torch
import torch.nn as nn

from rxnrep.model.gin import GINConv, GINConvGlobal
from tests.utils import create_graph_CO2


def test_gin_conv_CO2():
    """
    Test the feature update by setting weight matrix to identify, turning off bias,
    activation, residual, batch norm.


    CO2 with 3 atoms, 2 bonds, 1 global nodes and features:

     atom_feat:
        [[0,1,2,3],
         [4,5,6,7],
         [8,9,10,11]]

    bond_feat:
        [[0,1,2,3],
         [0,1,2,3],
         [4,5,6,7],
         [4,5,6,7],
    Note [0,1,2,3] corresponds to two edges for the same bond.


    global_feat:
        [[0,1,2,3]]


            /  v  \
          /    |    \
        /      |      \
    0[0] --- C[1] ---- O[2]
    """

    conv = init_net()

    g = create_graph_CO2(num_global_nodes=1, feat_dim=4)

    feats = {"atom": g.nodes["atom"].data["feat"], "bond": g.edges["bond"].data["feat"]}

    updated_feats = conv(g, feats)

    a = feats["atom"]
    b = feats["bond"]

    W = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )

    # bond feats
    ref_bond = torch.cat(
        (torch.stack([a[0] + a[1], a[1] + a[2]]), torch.stack([b[0], b[2]])), dim=-1
    )
    ref_bond = torch.mm(ref_bond, W.T)
    ref_bond = torch.repeat_interleave(ref_bond, 2, dim=0)

    assert torch.equal(updated_feats["bond"], ref_bond)

    # atom feats
    a = feats["atom"]
    b = updated_feats["bond"]

    sum_h = torch.stack((a[1], a[0] + a[2], a[1])) + a
    sum_e = torch.stack((b[0], b[0] + b[2], b[2]))
    ref_atom = torch.cat((sum_h, sum_e), dim=-1)
    ref_atom = torch.mm(ref_atom, W.T)

    assert torch.equal(updated_feats["atom"], ref_atom)


def test_gin_conv_global_CO2():
    """
    Test the feature update by setting weight matrix to identify, turning off bias,
    activation, residual, batch norm.


    CO2 with 3 atoms, 2 bonds, 1 global nodes and features:

     atom_feat:
        [[0,1,2,3],
         [4,5,6,7],
         [8,9,10,11]]

    bond_feat:
        [[0,1,2,3],
         [0,1,2,3],
         [4,5,6,7],
         [4,5,6,7],
    Note [0,1,2,3] corresponds to two edges for the same bond.


    global_feat:
        [[0,1,2,3]]


            /  v  \
          /    |    \
        /      |      \
    0[0] --- C[1] ---- O[2]
    """

    conv = init_net(has_global=True)

    g = create_graph_CO2(num_global_nodes=1, feat_dim=4)

    feats = {
        "atom": g.nodes["atom"].data["feat"],
        "bond": g.edges["bond"].data["feat"],
        "global": g.nodes["global"].data["feat"],
    }

    updated_feats = conv(g, feats)

    a = feats["atom"]
    b = feats["bond"]
    u = feats["global"]

    W = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )

    # bond feats
    ref_bond = torch.cat(
        (
            torch.stack([a[0] + a[1], a[1] + a[2]]),
            torch.stack([b[0], b[2]]),
            torch.stack([u[0], u[0]]),
        ),
        dim=-1,
    )
    ref_bond = torch.mm(ref_bond, W.T)
    ref_bond = torch.repeat_interleave(ref_bond, 2, dim=0)

    assert torch.equal(updated_feats["bond"], ref_bond)

    # atom feats
    a = feats["atom"]
    b = updated_feats["bond"]

    sum_h = torch.stack((a[1], a[0] + a[2], a[1])) + a
    sum_e = torch.stack((b[0], b[0] + b[2], b[2]))
    glb = torch.stack((u[0], u[0], u[0]))
    ref_atom = torch.cat((sum_h, sum_e, glb), dim=-1)
    ref_atom = torch.mm(ref_atom, W.T)

    assert torch.equal(updated_feats["atom"], ref_atom)

    # global feats
    a = updated_feats["atom"]
    b = updated_feats["bond"]

    sum_h = torch.sum(a, dim=0, keepdim=True)
    sum_e = 0.5 * torch.sum(b, dim=0, keepdim=True)
    ref_global = torch.cat((sum_h, sum_e, u), dim=-1)
    ref_global = torch.mm(ref_global, W.T)

    assert torch.equal(updated_feats["global"], ref_global)


def init_net(has_global=False):
    def set_identify_mapping(layer):
        """
        Set weight to identity and bias to zero.

        weight:

        [[1., 0., 0., 0., 1., 0., 0., 0.],
         [0., 1., 0., 0., 0., 1., 0., 0.],
         [0., 0., 1., 0., 0., 0., 1., 0.],
         [0., 0., 0., 1., 0., 0., 0., 1.]]

        when has_global is True:
        [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
         [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]

        """
        for name, param in layer.named_parameters():
            if "weight" in name:
                for i in range(4):
                    param[i, :] = 0
                    param[i, i] = 1
                    param[i, i + 4] = 1
                    if has_global:
                        param[i, i + 8] = 1
            elif "bias" in name:
                nn.init.zeros_(param)

    if has_global:
        conv_class = GINConvGlobal
    else:
        conv_class = GINConv
    conv = conv_class(
        input_dim=4,
        output_dim=4,
        num_fc_layers=1,
        batch_norm=False,
        activation=nn.Identity(),
    )

    # set MLP to identify mapping, i.e. identify weight matrix, zero bias
    set_identify_mapping(conv.mlp_atom)
    set_identify_mapping(conv.mlp_bond)
    if has_global:
        set_identify_mapping(conv.mlp_global)

    return conv
