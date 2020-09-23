from collections import defaultdict
from rdkit import Chem
import torch
from rxnrep.data.grapher import (
    create_hetero_molecule_graph,
    get_num_bond_nodes_information,
    get_bond_node_reorder_map_number,
    combine_graphs,
)


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
    def assert_graph(self_loop):

        m = Chem.MolFromSmiles("O=C=O")
        g = create_hetero_molecule_graph(m, self_loop)

        # number of atoms
        na = 3
        # number of bonds
        nb = 2
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

        ref_b2a_map = {0: [0, 1], 1: [1, 2]}
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

    assert_graph(False)
    assert_graph(True)


def test_get_num_bond_nodes_information():
    def assert_one(map_number, ref_unchanged, ref_changed, ref_artificial):
        unchanged, changed, artificial = get_num_bond_nodes_information(map_number)
        assert unchanged == ref_unchanged
        assert changed == ref_changed
        assert artificial == ref_artificial

    # case 1: all bonds are mapped
    assert_one([[2, 1], [0, 3]], 4, 0, 0)

    # case 2 some are not mapped
    assert_one([[2, 1, None], [0, 3, None], [None]], 4, 3, 0)

    # case 3 some has no bond
    assert_one([[2, 1, None], [0, 3], []], 4, 1, 1)
    assert_one([[2, 1, None], [], [0, None, 3], []], 4, 2, 2)


def test_get_bond_node_reorder_map_number():

    # case 1: all bonds are mapped
    bond_map_number = [[2, 1], [0, 3]]
    reorder = get_bond_node_reorder_map_number(bond_map_number)
    assert reorder == bond_map_number

    # case 2 some are not mapped
    bond_map_number = [[2, 1, None], [0, 3, None], [None]]
    reorder = get_bond_node_reorder_map_number(bond_map_number)
    assert reorder == [[2, 1, 4], [0, 3, 5], [6]]

    # case 3 some has no bond
    bond_map_number = [[2, 1, None], [], [0, None, 3], []]
    reorder = get_bond_node_reorder_map_number(bond_map_number)
    assert reorder == [[2, 1, 4], [6], [0, 5, 3], [7]]


def test_combine_graphs():
    def assert_graph_struct(self_loop):
        m = Chem.MolFromSmiles("O=C=O")
        na = 3  # num atoms
        nb = 2  # num bonds
        n_graph = 3  # number of graphs to combine to create new graph

        na = na * n_graph  # num atoms in new graph
        nb = nb * n_graph  # num bonds in new graph
        ne = 2 * nb  # num of edges between atoms and bonds in new graph

        g = create_hetero_molecule_graph(m, self_loop)
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


def test_combine_graph_feature():
    g1 = create_hetero_graph_CO2()
    g2 = create_hetero_graph_CO2()
    g3 = create_hetero_graph_CO2()

    # see `assert_graph_struct()` for how the bond map number is obtained
    atom_map_number = [[1, 4, 8], [0, 2, 7], [3, 5, 6]]
    bond_map_number = [[1, 4], [0, 2], [3, 5]]
    graph = combine_graphs([g1, g2, g3], atom_map_number, bond_map_number)

    a = torch.arange(12).reshape(3, 4)
    b = torch.arange(6).reshape(2, 3)
    g = torch.arange(2).reshape(1, 2)
    ref_atom_feats = torch.cat([a, a, a])[[3, 0, 4, 6, 1, 7, 8, 5, 2]]
    ref_bond_feats = torch.cat([b, b, b])[[2, 0, 3, 4, 1, 5]]
    ref_global_feats = torch.cat([g, g, g])

    assert torch.equal(graph.nodes["atom"].data["feat"], ref_atom_feats)
    assert torch.equal(graph.nodes["bond"].data["feat"], ref_bond_feats)
    assert torch.equal(graph.nodes["global"].data["feat"], ref_global_feats)
