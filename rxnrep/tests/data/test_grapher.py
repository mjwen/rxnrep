import functools
from collections import defaultdict

import numpy as np
import torch
from rdkit import Chem

from rxnrep.core.reaction import smiles_to_reaction
from rxnrep.data.dataset import build_hetero_graph_and_featurize_one_reaction
from rxnrep.data.featurizer import AtomFeaturizer, BondFeaturizer, GlobalFeaturizer
from rxnrep.data.grapher import (
    AtomTypeFeatureMasker,
    combine_graphs,
    create_hetero_molecule_graph,
    create_reaction_graph,
    get_atom_distance_to_reaction_center,
    get_bond_distance_to_reaction_center,
)


def create_hetero_graph_C(self_loop=False):
    """
    Create a single atom molecule C.

    atom_feats:
        [[0,1,2,3]]

    bond_feats:
        None

    global_feats:
        [[0,1]]
    """
    m = Chem.MolFromSmiles("[C]")
    feats = {
        "atom": torch.arange(4).reshape(1, 4),
        "bond": torch.tensor([], dtype=torch.int32).reshape(0, 3),
        "global": torch.arange(2).reshape(1, 2),
    }

    g = create_hetero_molecule_graph(m, self_loop)

    for ntype, ft in feats.items():
        g.nodes[ntype].data.update({"feat": ft})

    return g


def create_hetero_graph_CO2(self_loop=False):
    """
    Create a CO2 and add features.

    Molecule:
          0        1
    O(0) --- C(1) --- O(2)

    atom_feat:
        [[0,1,2,3],
         [4,5,6,7],
         [8,9,10,11]]

    bond_feat:
        [[0,1,2],
        [[3,4,5]]

    global_feat:
        [[0,1]]

    """
    m = Chem.MolFromSmiles("O=C=O")
    feats = {
        "atom": torch.arange(12).reshape(3, 4),
        "bond": torch.arange(6).reshape(2, 3),
        "global": torch.arange(2).reshape(1, 2),
    }

    g = create_hetero_molecule_graph(m, self_loop)

    for ntype, ft in feats.items():
        g.nodes[ntype].data.update({"feat": ft})

    return g


def get_bond_to_atom_map(g):
    """
    Query which atoms are associated with the bonds.

    Args:
        g: dgl graph

    Returns:
        dict: with bond index as the key and a tuple of atom indices of atoms that
            form the bond.
    """
    nbonds = g.number_of_nodes("bond")
    bond_to_atom_map = dict()
    for i in range(nbonds):
        atoms = g.successors(i, "b2a")
        bond_to_atom_map[i] = sorted(atoms)
    return bond_to_atom_map


def get_atom_to_bond_map(g):
    """
    Query which bonds are associated with the atoms.

    Args:
        g: dgl graph

    Returns:
        dict: with atom index as the key and a list of indices of bonds is
        connected to the atom.
    """
    natoms = g.number_of_nodes("atom")
    atom_to_bond_map = dict()
    for i in range(natoms):
        bonds = g.successors(i, "a2b")
        if len(bonds) > 0:
            atom_to_bond_map[i] = sorted(list(bonds))
    return atom_to_bond_map


def get_hetero_self_loop_map(g, ntype):
    num = g.number_of_nodes(ntype)
    if ntype == "atom":
        etype = "a2a"
    elif ntype == "bond":
        etype = "b2b"
    elif ntype == "global":
        etype = "g2g"
    else:
        raise ValueError("not supported node type: {}".format(ntype))
    self_loop_map = dict()
    for i in range(num):
        suc = g.successors(i, etype)
        self_loop_map[i] = list(suc)

    return self_loop_map


def test_create_hetero_molecule_graph():
    def assert_graph(m, ref_b2a_map, self_loop):
        g = create_hetero_molecule_graph(m, self_loop)

        # number of atoms
        na = m.GetNumAtoms()
        # number of bonds
        nb = m.GetNumBonds()
        # number of edges between atoms and bonds
        ne = 2 * nb

        nodes = ["atom", "bond", "global"]
        num_nodes = [g.number_of_nodes(n) for n in nodes]
        ref_num_nodes = [na, nb, 1]

        if self_loop:
            edges = ["a2b", "b2a", "a2g", "g2a", "b2g", "g2b", "a2a", "b2b", "g2g"]
            num_edges = [g.number_of_edges(e) for e in edges]
            ref_num_edges = [ne, ne, na, na, nb, nb, na, nb, 1]

        else:
            edges = ["a2b", "b2a", "a2g", "g2a", "b2g", "g2b"]
            num_edges = [g.number_of_edges(e) for e in edges]
            ref_num_edges = [ne, ne, na, na, nb, nb]

        assert set(g.ntypes) == set(nodes)
        assert set(g.etypes) == set(edges)
        assert num_nodes == ref_num_nodes
        assert num_edges == ref_num_edges

        ref_a2b_map = defaultdict(list)
        for b, atoms in ref_b2a_map.items():
            for a in atoms:
                ref_a2b_map[a].append(b)
        ref_a2b_map = {a: sorted(bonds) for a, bonds in ref_a2b_map.items()}

        b2a_map = get_bond_to_atom_map(g)
        a2b_map = get_atom_to_bond_map(g)
        assert ref_b2a_map == b2a_map
        assert ref_a2b_map == a2b_map

        if self_loop:
            for nt, n in zip(nodes, num_nodes):
                assert get_hetero_self_loop_map(g, nt) == {i: [i] for i in range(n)}

    m = Chem.MolFromSmiles("O=C=O")
    ref_b2a_map = {0: [0, 1], 1: [1, 2]}
    assert_graph(m, ref_b2a_map, False)
    assert_graph(m, ref_b2a_map, True)

    # test single atom molecules (no bonds)
    m = Chem.MolFromSmiles("[C]")
    ref_b2a_map = {}
    assert_graph(m, ref_b2a_map, False)
    assert_graph(m, ref_b2a_map, True)


def test_combine_graphs_CO2():
    def assert_graph_struct(self_loop):
        g = create_hetero_graph_CO2(self_loop)
        na = 3  # num atoms
        nb = 2  # num bonds
        n_graph = 3  # number of graphs to combine to create new graph

        na = na * n_graph  # num atoms in new graph
        nb = nb * n_graph  # num bonds in new graph
        ne = 2 * nb  # num of edges between atoms and bonds in new graph

        atom_map_number = [[1, 4, 8], [0, 2, 7], [3, 5, 6]]
        #
        # Give the atom_map_number, bonds would be
        # [[(1,4), (4,8)], [(0,2), (2,7)], [(3,5), (5,6)]],
        # If we order them, the order would be
        # [[1, 4], [0,2], [3,5]]
        #
        bond_map_number = [[1, 4], [0, 2], [3, 5]]
        g = combine_graphs([g] * n_graph, atom_map_number, bond_map_number)

        nodes = ["atom", "bond", "global"]
        num_nodes = [g.number_of_nodes(n) for n in nodes]
        ref_num_nodes = [na, nb, n_graph]

        if self_loop:
            edges = ["a2b", "b2a", "a2g", "g2a", "b2g", "g2b", "a2a", "b2b", "g2g"]
            num_edges = [g.number_of_edges(e) for e in edges]
            ref_num_edges = [ne, ne, na, na, nb, nb, na, nb, n_graph]

        else:
            edges = ["a2b", "b2a", "a2g", "g2a", "b2g", "g2b"]
            num_edges = [g.number_of_edges(e) for e in edges]
            ref_num_edges = [ne, ne, na, na, nb, nb]

        assert set(g.ntypes) == set(nodes)
        assert set(g.etypes) == set(edges)
        assert num_nodes == ref_num_nodes
        assert num_edges == ref_num_edges

        # atom map number: [[1, 4, 8], [0, 2, 7], [3, 5, 6]]
        # b2a map for a single molecule {0: [0, 1], 1: [1, 2]}, meaning bond 0 is
        # connected atoms 0 and 1, bond 1 is connected to atoms 1 and 2.
        # Then the b2a map for all molecules is:
        # {0: [1, 4], 1: [4, 8], 2: [0, 2], 3: [2, 7], 4: [3, 5], 5: [5, 6]}
        # considering the bond map number:  [[1, 4], [0, 2], [3, 5]]
        # the b2a map is then:
        ref_b2a_map = {1: [1, 4], 4: [4, 8], 0: [0, 2], 2: [2, 7], 3: [3, 5], 5: [5, 6]}

        ref_a2b_map = defaultdict(list)
        for b, atoms in ref_b2a_map.items():
            for a in atoms:
                ref_a2b_map[a].append(b)
        ref_a2b_map = {a: sorted(bonds) for a, bonds in ref_a2b_map.items()}

        b2a_map = get_bond_to_atom_map(g)
        a2b_map = get_atom_to_bond_map(g)
        assert ref_b2a_map == b2a_map
        assert ref_a2b_map == a2b_map

        if self_loop:
            for nt, n in zip(nodes, num_nodes):
                assert get_hetero_self_loop_map(g, nt) == {i: [i] for i in range(n)}

    assert_graph_struct(False)
    assert_graph_struct(True)


def test_combine_graphs_C():
    """Single atom molecule without bond nodes."""

    def assert_graph_struct(self_loop):
        g = create_hetero_graph_C(self_loop)
        na = 1  # num atoms
        nb = 0  # num bonds
        n_graph = 3  # number of graphs to combine to create new graph

        na = na * n_graph  # num atoms in new graph
        nb = nb * n_graph  # num bonds in new graph
        ne = 2 * nb  # num of edges between atoms and bonds in new graph

        atom_map_number = [[1], [0], [2]]
        bond_map_number = [[], [], []]
        g = combine_graphs([g] * n_graph, atom_map_number, bond_map_number)

        nodes = ["atom", "bond", "global"]
        num_nodes = [g.number_of_nodes(n) for n in nodes]
        ref_num_nodes = [na, nb, n_graph]

        if self_loop:
            edges = ["a2b", "b2a", "a2g", "g2a", "b2g", "g2b", "a2a", "b2b", "g2g"]
            num_edges = [g.number_of_edges(e) for e in edges]
            ref_num_edges = [ne, ne, na, na, nb, nb, na, nb, n_graph]

        else:
            edges = ["a2b", "b2a", "a2g", "g2a", "b2g", "g2b"]
            num_edges = [g.number_of_edges(e) for e in edges]
            ref_num_edges = [ne, ne, na, na, nb, nb]

        assert set(g.ntypes) == set(nodes)
        assert set(g.etypes) == set(edges)
        assert num_nodes == ref_num_nodes
        assert num_edges == ref_num_edges

        ref_b2a_map = {}

        ref_a2b_map = defaultdict(list)
        for b, atoms in ref_b2a_map.items():
            for a in atoms:
                ref_a2b_map[a].append(b)
        ref_a2b_map = {a: sorted(bonds) for a, bonds in ref_a2b_map.items()}

        b2a_map = get_bond_to_atom_map(g)
        a2b_map = get_atom_to_bond_map(g)
        assert ref_b2a_map == b2a_map
        assert ref_a2b_map == a2b_map

        if self_loop:
            for nt, n in zip(nodes, num_nodes):
                assert get_hetero_self_loop_map(g, nt) == {i: [i] for i in range(n)}

    assert_graph_struct(False)
    assert_graph_struct(True)


def test_combine_graph_feature():
    """
    Two CO2 and one C molecules: no bond features.
    """
    g1 = create_hetero_graph_CO2()
    g2 = create_hetero_graph_CO2()
    g3 = create_hetero_graph_C()

    # see `assert_graph_struct()` for how the bond map number is obtained
    atom_map_number = [[1, 4, 6], [0, 2, 5], [3]]
    bond_map_number = [[1, 3], [0, 2], []]
    graph = combine_graphs([g1, g2, g3], atom_map_number, bond_map_number)

    a1 = torch.arange(12).reshape(3, 4)
    b1 = torch.arange(6).reshape(2, 3)
    g1 = torch.arange(2).reshape(1, 2)
    a2 = torch.arange(4).reshape(1, 4)
    b2 = torch.arange(0).reshape(0, 3)
    g2 = torch.arange(2).reshape(1, 2)
    ref_atom_feats = torch.cat([a1, a1, a2])[[3, 0, 4, 6, 1, 5, 2]]
    ref_bond_feats = torch.cat([b1, b1, b2])[[2, 0, 3, 1]]
    ref_global_feats = torch.cat([g1, g1, g2])

    assert torch.equal(graph.nodes["atom"].data["feat"], ref_atom_feats)
    assert torch.equal(graph.nodes["bond"].data["feat"], ref_bond_feats)
    assert torch.equal(graph.nodes["global"].data["feat"], ref_global_feats)


def test_combine_graph_feature_2():
    """
    Two C molecules: no bond features.

    The combined graph should have a bond feature of shape (0, D), where D is is
    feature size.
    """
    g1 = create_hetero_graph_C()
    g2 = create_hetero_graph_C()

    atom_map_number = [[1], [0]]
    bond_map_number = [[], []]
    graph = combine_graphs([g1, g2], atom_map_number, bond_map_number)

    a = torch.arange(4).reshape(1, 4)
    b = torch.arange(0).reshape(0, 3).type(torch.int32)
    g = torch.arange(2).reshape(1, 2)
    ref_atom_feats = torch.cat([a, a])[[1, 0]]
    ref_bond_feats = torch.cat([b, b])
    ref_global_feats = torch.cat([g, g])

    assert torch.equal(graph.nodes["atom"].data["feat"], ref_atom_feats)
    assert torch.equal(graph.nodes["bond"].data["feat"], ref_bond_feats)
    assert torch.equal(graph.nodes["global"].data["feat"], ref_global_feats)


def test_create_reaction_graph():
    """Create a reaction: CH3CH2+ + CH3CH2CH2 --> CH3 + CH3CH2CH2CH2+

    m1:
        2
    C2-----C0

    m2:
        0      1
    C1-----C3-----C4

    m3:

    C2

    m4:
        0      1      2
    C1-----C3-----C4-----C0

    combined union graph (the bond index is assigned to mimic combine_graphs() function)

              2
          C2-----C0
                  |
                  |  3
        0      1  |
    C1-----C3-----C4
    """

    m1 = Chem.MolFromSmiles("[CH3:3][CH2+:1]")
    m2 = Chem.MolFromSmiles("[CH3:2][CH2:4][CH2:5]")
    m3 = Chem.MolFromSmiles("[CH3:3]")
    m4 = Chem.MolFromSmiles("[CH3:2][CH2:4][CH2:5][CH2+:1]")

    g1 = create_hetero_molecule_graph(m1)
    g2 = create_hetero_molecule_graph(m2)
    g3 = create_hetero_molecule_graph(m3)
    g4 = create_hetero_molecule_graph(m4)

    reactant_atom_map = [[2, 0], [1, 3, 4]]
    reactant_bond_map = [[2], [0, 1]]
    product_atom_map = [[2], [1, 3, 4, 0]]
    product_bond_map = [[], [0, 1, 2]]

    reactant = combine_graphs([g1, g2], reactant_atom_map, reactant_bond_map)
    product = combine_graphs([g3, g4], product_atom_map, product_bond_map)

    g = create_reaction_graph(
        reactant, product, num_unchanged_bonds=2, num_lost_bonds=1, num_added_bonds=1
    )

    b2a = get_bond_to_atom_map(g)
    a2b = get_atom_to_bond_map(g)

    # reference bond to atom and atom to bond map
    ref_b2a_map = {0: [1, 3], 1: [3, 4], 2: [0, 2], 3: [0, 4]}
    ref_a2b_map = defaultdict(list)
    for b, atoms in ref_b2a_map.items():
        for a in atoms:
            ref_a2b_map[a].append(b)
    ref_a2b_map = {a: sorted(bonds) for a, bonds in ref_a2b_map.items()}

    assert b2a == ref_b2a_map
    assert a2b == ref_a2b_map


def test_get_atom_bond_distance_to_reaction_center():
    r"""Create a reaction: CH3CH2+ + CH3CH2CH2 --> CH3 + CH3CH2CH2CH2+

    m1:
        3
    C2-----C0

    m2:
        0      2
    C1-----C3-----C4
     \     |
     1 \   |  3
         \ C5

    m3:

    C2

    m4:
        0      2      4
    C1-----C3-----C4-----C0
     \     |
     1 \   |  3
         \ C5


    combined union graph (the bond index is assigned to mimic combine_graphs() function)

              4
          C2-----C0
                  |
                  |  5
        0      2  |
    C1-----C3-----C4
     \     |
     1 \   |  3
         \ C5

    atom hop distances:
    [max_hop + 2, 2, 0, 1, max_hop + 1, 2]

    bond hop distances:
    [2, 3, 1, 2, 0, max_hop + 1]
    """
    smi_rxn = "[CH3:3][CH2+:1].[CH2:2]1[CH1:4]([CH2:6]1)[CH2:5]>>[CH3:3].[CH2:2]1[CH1:4]([CH2:6]1)[CH2:5][CH2+:1]"
    rxn = smiles_to_reaction(smi_rxn)

    max_hop = 1
    atom_hop_dist = get_atom_distance_to_reaction_center(rxn, max_hop)
    assert atom_hop_dist == [max_hop + 2, 1, 0, 1, max_hop + 1, 1]
    bond_hop_dist = get_bond_distance_to_reaction_center(rxn, atom_hop_dist, max_hop)
    assert bond_hop_dist == [1, 1, 1, 1, 0, max_hop + 1]

    max_hop = 2
    atom_hop_dist = get_atom_distance_to_reaction_center(rxn, max_hop)
    assert atom_hop_dist == [max_hop + 2, 2, 0, 1, max_hop + 1, 2]
    bond_hop_dist = get_bond_distance_to_reaction_center(rxn, atom_hop_dist, max_hop)
    assert bond_hop_dist == [2, 2, 1, 2, 0, max_hop + 1]

    max_hop = 3
    atom_hop_dist = get_atom_distance_to_reaction_center(rxn, max_hop)
    assert atom_hop_dist == [max_hop + 2, 2, 0, 1, max_hop + 1, 2]
    bond_hop_dist = get_bond_distance_to_reaction_center(rxn, atom_hop_dist, max_hop)
    assert bond_hop_dist == [2, 3, 1, 2, 0, max_hop + 1]

    # 2 reaction centers in two disjoint part of the reaction graph
    smi_rxn = "[Cl:1][CH:13]([Cl:12])[Cl:14].[OH:9][CH3:15].[c:2]1([CH3:3])[cH:4][cH:5][c:6]([C:7]#[N:8])[cH:10][cH:11]1>>[CH4:15].[Cl:12][CH2:13][Cl:14].[ClH:1].[c:2]1([CH3:3])[cH:4][cH:5][c:6]([C:7](=[NH:8])[OH:9])[cH:10][cH:11]1"
    rxn = smiles_to_reaction(smi_rxn)

    max_hop = 3
    atom_hop_dist = get_atom_distance_to_reaction_center(rxn, max_hop)
    assert atom_hop_dist == [0, 3, 3, 3, 2, 1, 4, 1, 5, 2, 3, 1, 0, 1, 0]
    bond_hop_dist = get_bond_distance_to_reaction_center(rxn, atom_hop_dist, max_hop)
    assert bond_hop_dist == [3, 3, 3, 3, 2, 1, 2, 1, 3, 1, 1, 0, 0, 4]


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
