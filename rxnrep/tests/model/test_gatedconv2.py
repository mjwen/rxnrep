import torch
import torch.nn as nn

from rxnrep.core.molecule import Molecule
from rxnrep.data.to_graph import mol_to_graph
from rxnrep.model.gatedconv2 import GatedGCNConv


def create_graph_CO2():
    """
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
    n_v = 1
    n_a = 3
    n_b = 2
    feat_size = 4

    smi = "O=C=O"
    m = Molecule.from_smiles(smi)
    g = mol_to_graph(m, num_global_nodes=n_v)

    g.nodes["atom"].data.update(
        {"feat": torch.arange(n_a * feat_size).float().reshape(n_a, feat_size)}
    )

    # this create zero tensor (i.e. of shape (0, 3)) if n_b == 0
    bond_feats = torch.arange(n_b * feat_size).float().reshape(n_b, feat_size)
    bond_feats = torch.repeat_interleave(bond_feats, 2, dim=0)
    g.edges["bond"].data.update({"feat": bond_feats})

    g.nodes["global"].data.update(
        {"feat": torch.arange(n_v * feat_size).float().reshape(n_v, feat_size)}
    )

    return g


def test_gated_gcn_conv():
    """
    Test the feature update by setting weight matrix to identify, turning off bias,
    activation, residual, batch norm.
    """

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

    # we modify

    g = create_graph_CO2()

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
