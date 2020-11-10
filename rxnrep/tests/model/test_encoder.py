import torch
import numpy as np
from rxnrep.model.encoder import create_reaction_features


def test_create_reaction_features():
    """
    We assume two reactions (bond index inside parenthesis are its feature index):

      0(0)   1(1)       0(5)  1(6)
    C----C  C----C  -> C----C----C + C
      m0       m1        m3          m4

     0(2) 2(4)  1(3)       0(7)     1(8)
    C----C----C----C  -> C----C + C----C
          m2               m5       m6

    """
    metadata = {
        "reactant_num_molecules": [2, 1],
        "product_num_molecules": [2, 2],
        "num_unchanged_bonds": [1, 2],
        "num_lost_bonds": [1, 1],
        "num_added_bonds": [1, 0],
    }

    # feats
    atom_feats = torch.from_numpy(np.arange(16 * 2).reshape(16, 2).astype(np.float32))
    bond_feats = torch.from_numpy(np.arange(9 * 3).reshape(9, 3).astype(np.float32))
    global_feats = torch.from_numpy(np.arange(7 * 4).reshape(7, 4).astype(np.float32))
    ref_atom_feats = (
        atom_feats[[8, 9, 10, 11, 12, 13, 14, 15]]
        - atom_feats[[0, 1, 2, 3, 4, 5, 6, 7]]
    )
    ref_bond_feats = torch.stack(
        [
            bond_feats[5] - bond_feats[0],
            -bond_feats[1],
            bond_feats[6],
            bond_feats[7] - bond_feats[2],
            bond_feats[8] - bond_feats[3],
            -bond_feats[4],
        ]
    )

    ref_global_feats = torch.stack(
        [
            torch.mean(global_feats[3:5], dim=0) - torch.mean(global_feats[0:2], dim=0),
            torch.mean(global_feats[5:7], dim=0) - global_feats[2],
        ]
    )

    molecule_feats = {"atom": atom_feats, "bond": bond_feats, "global": global_feats}
    diff_feats = create_reaction_features(molecule_feats, metadata)

    assert torch.equal(diff_feats["atom"], ref_atom_feats)
    assert torch.equal(diff_feats["bond"], ref_bond_feats)
    assert torch.equal(diff_feats["global"], ref_global_feats)
