import functools

import numpy as np

from rxnrep.core.reaction import smiles_to_reaction
from rxnrep.data.augmentation import AtomTypeFeatureMasker
from rxnrep.data.dataset import build_hetero_graph_and_featurize_one_reaction
from rxnrep.data.featurizer import AtomFeaturizer, BondFeaturizer, GlobalFeaturizer


def test_atom_type_masker():
    smi_rxn = "[Cl:1][CH:13]([Cl:12])[Cl:14].[OH:9][CH3:15].[c:2]1([CH3:3])[cH:4][cH:5][c:6]([C:7]#[N:8])[cH:10][cH:11]1>>[CH4:15].[Cl:12][CH2:13][Cl:14].[ClH:1].[c:2]1([CH3:3])[cH:4][cH:5][c:6]([C:7](=[NH:8])[OH:9])[cH:10][cH:11]1"
    rxn = smiles_to_reaction(smi_rxn)

    species = ["C", "O", "Cl", "N"]
    atom_featurizer = AtomFeaturizer()
    bond_featurizer = BondFeaturizer()
    global_featurizer = GlobalFeaturizer()

    reactants_g, products_g, _ = build_hetero_graph_and_featurize_one_reaction(
        rxn,
        functools.partial(atom_featurizer, allowable_atom_type=species),
        bond_featurizer,
        global_featurizer,
    )

    # set random seed since we used it in masker
    np.random.seed(5)

    feature_mean = np.array([2.0, 3.0, 4.0, 5.0])
    feature_std = np.array([1.0, 2.0, 3.0, 4.0])
    masker = AtomTypeFeatureMasker(
        allowable_types=species,
        feature_name=atom_featurizer.feature_name,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    reactants_g, products_g, is_atom_masked, masked_atom_labels = masker.mask_features(
        reactants_g, products_g, rxn
    )
    masked_atoms = [i for i, b in enumerate(is_atom_masked) if b]

    # atoms 1, 5, and 7 are masked, of species C, C, and N.
    # Sorted species are ['C', 'Cl', 'N', 'O'], so the labels for the masked atoms are
    # 0, 0, 2
    assert is_atom_masked == [
        False,
        True,
        False,
        False,
        False,
        True,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    assert masked_atom_labels == [0, 0, 2]

    reactants_atom_feats = reactants_g.nodes["atom"].data["feat"]
    products_atom_feats = products_g.nodes["atom"].data["feat"]

    masked_feats = -feature_mean / feature_std
    for atom in masked_atoms:
        np.array_equal(reactants_atom_feats[atom][:4], masked_feats)
        np.array_equal(products_atom_feats[atom][:4], masked_feats)

    #
    # test the case we do not provide feature mean and std
    #
    masker = AtomTypeFeatureMasker(
        allowable_types=species,
        feature_name=atom_featurizer.feature_name,
        feature_mean=None,
        feature_std=None,
    )
    reactants_g, products_g, is_atom_masked, masked_atom_labels = masker.mask_features(
        reactants_g, products_g, rxn
    )
    masked_atoms = [i for i, b in enumerate(is_atom_masked) if b]

    # atoms 3, 10, and 11 are masked, of species C, C, and Cl.
    # Sorted species are ['C', 'Cl', 'N', 'O'], so the labels for the masked atoms are
    # 0, 0, 1
    assert is_atom_masked == [
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        False,
        False,
        False,
    ]
    assert masked_atom_labels == [0, 0, 1]

    reactants_atom_feats = reactants_g.nodes["atom"].data["feat"]
    products_atom_feats = products_g.nodes["atom"].data["feat"]

    masked_feats = [0.0, 0.0, 0.0, 0.0]
    for atom in masked_atoms:
        np.array_equal(reactants_atom_feats[atom][:4], masked_feats)
        np.array_equal(products_atom_feats[atom][:4], masked_feats)
