import dgl
import torch
import torch.nn as nn
from dgl.ops import segment_reduce

from rxnrep.model.gatedconv2 import GatedGCNConv
from rxnrep.tests.utils import create_graph_C, create_graph_CO2


def test_gated_gcn_conv_C():
    """
    Test the feature update by setting weight matrix to identify, turning off bias,
    activation, residual, batch norm.


    C with 1 atoms, 0 bonds, 1 global nodes and features:

     atom_feat:
        [[0,1,2,3]]

    bond_feat:
        [[]]


    global_feat:
        [[0,1,2,3]]
    """
    conv = init_net()

    g = create_graph_C(num_global_nodes=1, feat_dim=4)

    feats = {
        "atom": g.nodes["atom"].data["feat"],
        "bond": g.edges["bond"].data["feat"],
        "global": g.nodes["global"].data["feat"],
    }

    updated_feats = conv(g, feats)

    a = feats["atom"]
    b = feats["bond"]
    v = feats["global"]

    # bond feats
    ref_bond = b
    assert torch.equal(updated_feats["bond"], ref_bond)

    # atom feats
    ref_atom = a + v
    assert torch.equal(updated_feats["atom"], ref_atom)

    # global feats
    ref_global = ref_atom + v
    assert torch.equal(updated_feats["global"], ref_global)


def test_gated_gcn_conv_CO2():
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

    feats = {
        "atom": g.nodes["atom"].data["feat"],
        "bond": g.edges["bond"].data["feat"],
        "global": g.nodes["global"].data["feat"],
    }

    updated_feats = conv(g, feats)

    a = feats["atom"]
    b = feats["bond"]
    v = feats["global"]

    # bond feats
    ref_bond = (
        torch.stack([a[0] + a[1], a[1] + a[2]])  # atom
        + torch.stack([b[0], b[2]])  # bond
        + v[0]  # global
    )
    ref_bond = torch.repeat_interleave(ref_bond, 2, dim=0)
    assert torch.equal(updated_feats["bond"], ref_bond)

    # atom feats
    a = feats["atom"]
    b = updated_feats["bond"]
    v = feats["global"]

    sigma = torch.sigmoid(b)
    sigma = [sigma[0], sigma[2]]  # sigma for bond 0 and 1
    sigma_times_h = torch.stack(
        [
            a[1] * sigma[0] / (sigma[0] + 1e-6),
            (a[0] * sigma[0] + a[2] * sigma[1]) / (sigma[0] + sigma[1] + 1e-6),
            a[1] * sigma[1] / (sigma[1] + 1e-6),
        ]
    )

    ref_atom = a + sigma_times_h + v
    assert torch.equal(updated_feats["atom"], ref_atom)

    # global feats
    a = updated_feats["atom"]
    b = updated_feats["bond"]
    v = feats["global"]

    ref_global = torch.mean(a, dim=0) + torch.mean(b, dim=0) + v
    assert torch.equal(updated_feats["global"], ref_global)


def test_gated_gcn_conv_C_CO2():
    """
    Test by batching C and CO2 together.
    """

    conv = init_net()

    g1 = create_graph_CO2(num_global_nodes=1, feat_dim=4)
    g2 = create_graph_C(num_global_nodes=1, feat_dim=4)
    g = dgl.batch([g1, g2])

    feats = {
        "atom": g.nodes["atom"].data["feat"],
        "bond": g.edges["bond"].data["feat"],
        "global": g.nodes["global"].data["feat"],
    }

    updated_feats = conv(g, feats)

    a = feats["atom"]
    b = feats["bond"]
    v = feats["global"]

    # bond feats
    ref_bond = (
        torch.stack([a[0] + a[1], a[1] + a[2]])  # atom
        + torch.stack([b[0], b[2]])  # bond
        + v[0]  # global
    )
    ref_bond = torch.repeat_interleave(ref_bond, 2, dim=0)
    assert torch.equal(updated_feats["bond"], ref_bond)

    # atom feats
    a = feats["atom"]
    b = updated_feats["bond"]
    v = feats["global"]

    # for CO2
    sigma = torch.sigmoid(b)
    sigma = [sigma[0], sigma[2]]  # sigma for bond 0 and 1
    sigma_times_h_1 = torch.stack(
        [
            a[1] * sigma[0] / (sigma[0] + 1e-6),
            (a[0] * sigma[0] + a[2] * sigma[1]) / (sigma[0] + sigma[1] + 1e-6),
            a[1] * sigma[1] / (sigma[1] + 1e-6),
        ]
    )
    # for C
    sigma_times_h_2 = torch.zeros(1, sigma_times_h_1.shape[1])
    # together
    sigma_times_h = torch.cat([sigma_times_h_1, sigma_times_h_2])

    # 3 atoms for CO2 and 1 atom for C
    v = torch.repeat_interleave(v, torch.tensor([3, 1]), dim=0)

    ref_atom = a + sigma_times_h + v
    assert torch.equal(updated_feats["atom"], ref_atom)

    # global feats
    a = updated_feats["atom"]
    b = updated_feats["bond"]
    v = feats["global"]

    # 3 atoms for CO2 and 1 atom for C
    # 2 bonds (2*2 edges) for CO2 and 0 bonds for C
    a = segment_reduce(torch.tensor([3, 1]), a, reducer="mean")
    b = segment_reduce(torch.tensor([2 * 2, 0 * 2]), b, reducer="mean")

    ref_global = a + b + v
    assert torch.equal(updated_feats["global"], ref_global)


def init_net():
    def set_identify_mapping(layer):
        """
        Set weight to identity and bias to zero.
        """
        for name, param in layer.named_parameters():
            if "weight" in name:
                nn.init.eye_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    conv = GatedGCNConv(
        input_dim=4, output_dim=4, batch_norm=False, activation=nn.Identity()
    )

    # set MLP to identify mapping, i.e. identify weight matrix, zero bias
    set_identify_mapping(conv.A)
    set_identify_mapping(conv.B)
    set_identify_mapping(conv.C)
    set_identify_mapping(conv.D)
    set_identify_mapping(conv.E)
    set_identify_mapping(conv.F)
    set_identify_mapping(conv.G)
    set_identify_mapping(conv.H)
    set_identify_mapping(conv.I)

    return conv
