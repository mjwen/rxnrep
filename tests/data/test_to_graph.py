import dgl
import numpy as np
import torch

from rxnrep.core.molecule import Molecule
from rxnrep.data.to_graph import combine_graphs, create_reaction_graph, mol_to_graph
from tests.utils import create_graph_C, create_graph_CO2


def test_create_graph():
    def assert_one(g, n_a, n_b, n_v):

        num_nodes = {"atom": n_a}
        num_edges = {"bond": 2 * n_b}

        if n_v > 0:
            num_nodes["global"] = n_v
            num_edges["g2a"] = n_a * n_v
            num_edges["a2g"] = n_a * n_v

        assert set(g.ntypes) == set(num_nodes.keys())
        assert set(g.etypes) == set(num_edges.keys())
        for k, n in num_nodes.items():
            assert g.number_of_nodes(k) == n
        for k, n in num_edges.items():
            assert g.number_of_edges(k) == n

    for n_v in range(3):
        g = create_graph_C(num_global_nodes=n_v)
        assert_one(g, 1, 0, n_v)

    for n_v in range(3):
        g = create_graph_CO2(num_global_nodes=n_v)
        assert_one(g, 3, 2, n_v)


def test_batch_graph():

    n_a_1 = 1
    n_b_1 = 0
    n_v_1 = 1
    g1 = create_graph_C(num_global_nodes=n_v_1)

    n_a_2 = 3
    n_b_2 = 2
    n_v_2 = 1
    g2 = create_graph_CO2(num_global_nodes=n_v_2)

    g = dgl.batch([g1, g2])

    num_nodes = {"atom": n_a_1 + n_a_2, "global": n_v_1 + n_v_2}
    num_edges = {
        "bond": 2 * (n_b_1 + n_b_2),
        "g2a": n_a_1 * n_v_1 + n_a_2 * n_v_2,
        "a2g": n_a_1 * n_v_1 + n_a_2 * n_v_2,
    }

    assert set(g.ntypes) == set(num_nodes.keys())
    assert set(g.etypes) == set(num_edges.keys())
    for k, n in num_nodes.items():
        assert g.number_of_nodes(k) == n
    for k, n in num_edges.items():
        assert g.number_of_edges(k) == n

    # assert features
    for t in num_nodes.keys():
        assert torch.equal(
            g.nodes[t].data["feat"],
            torch.cat([g1.nodes[t].data["feat"], g2.nodes[t].data["feat"]]),
        )

    for t in ["bond"]:
        assert torch.equal(
            g.edges[t].data["feat"],
            torch.cat([g1.edges[t].data["feat"], g2.edges[t].data["feat"]]),
        )


def test_combine_graphs_CO2():
    """
    Three CO2 mols.
    """

    for nv in [0, 1]:
        g = create_graph_CO2(num_global_nodes=nv)

        atom_map_number = [[1, 4, 8], [0, 2, 7], [3, 5, 6]]
        bond_map_number = [[1, 4], [0, 2], [3, 5]]
        # bond 1 is between atoms (1, 4), bond 4 is the between atoms (4, 8) ...
        bond_to_atom_map = {
            1: (1, 4),
            4: (4, 8),
            0: (0, 2),
            2: (2, 7),
            3: (3, 5),
            5: (5, 6),
        }

        if nv == 0:
            global_map_number = [[], [], []]
        elif nv == 1:
            global_map_number = [[0], [1], [2]]
        else:
            raise ValueError

        assert_combine_graphs(
            [g, g, g],
            [3, 3, 3],
            [2, 2, 2],
            [nv, nv, nv],
            atom_map_number,
            bond_map_number,
            global_map_number,
            bond_to_atom_map,
        )


def test_combine_graphs_C():
    """
    Three single atom molecule without bond.
    """

    for nv in [0, 1]:
        g = create_graph_C(num_global_nodes=nv)

        atom_map_number = [[1], [2], [0]]
        bond_map_number = [[], [], []]
        bond_to_atom_map = {}

        if nv == 0:
            global_map_number = [[], [], []]
        elif nv == 1:
            global_map_number = [[0], [1], [2]]
        else:
            raise ValueError

        assert_combine_graphs(
            [g, g, g],
            [1, 1, 1],
            [0, 0, 0],
            [nv, nv, nv],
            atom_map_number,
            bond_map_number,
            global_map_number,
            bond_to_atom_map,
        )


def test_combine_graphs_C_CO2():
    """
    CO2, C and CO2; and C, CO2 and C.
    """

    for nv in [0, 1]:
        g1 = create_graph_CO2(num_global_nodes=nv)
        g2 = create_graph_C(num_global_nodes=nv)

        if nv == 0:
            global_map_number = [[], [], []]
        elif nv == 1:
            global_map_number = [[0], [1], [2]]
        else:
            raise ValueError

        # CO2, C and CO2
        atom_map_number = [[1, 4, 5], [3], [0, 2, 6]]
        bond_map_number = [[1, 3], [], [0, 2]]
        # bond 1 is between atoms (1, 4), bond 3 is the between atoms (4, 5) ...
        bond_to_atom_map = {1: (1, 4), 3: (4, 5), 0: (0, 2), 2: (2, 6)}

        assert_combine_graphs(
            [g1, g2, g1],
            [3, 1, 3],
            [2, 0, 2],
            [nv, nv, nv],
            atom_map_number,
            bond_map_number,
            global_map_number,
            bond_to_atom_map,
        )

        # C, CO2 and C
        atom_map_number = [[3], [4, 0, 2], [1]]
        bond_map_number = [[], [1, 0], []]
        bond_to_atom_map = {1: (4, 0), 0: (0, 2)}

        assert_combine_graphs(
            [g2, g1, g2],
            [1, 3, 1],
            [0, 2, 0],
            [nv, nv, nv],
            atom_map_number,
            bond_map_number,
            global_map_number,
            bond_to_atom_map,
        )


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

    smiles = [
        "[CH3:3][CH2+:1]",
        "[CH3:2][CH2:4][CH2:5]",
        "[CH3:3]",
        "[CH3:2][CH2:4][CH2:5][CH2+:1]",
    ]

    reactant_atom_map = [[2, 0], [1, 3, 4]]
    reactant_bond_map = [[2], [0, 1]]
    product_atom_map = [[2], [1, 3, 4, 0]]
    product_bond_map = [[], [0, 1, 2]]

    for nv in [0, 1]:

        graphs = [
            mol_to_graph(Molecule.from_smiles(s), num_global_nodes=nv) for s in smiles
        ]

        reactant = combine_graphs(
            [graphs[0], graphs[1]], reactant_atom_map, reactant_bond_map
        )
        product = combine_graphs(
            [graphs[2], graphs[3]], product_atom_map, product_bond_map
        )

        g = create_reaction_graph(
            reactant,
            product,
            num_unchanged_bonds=2,
            num_lost_bonds=1,
            num_added_bonds=1,
            num_global_nodes=nv,
        )

        na = 5
        nb = 4
        assert_graph_quantity(
            g,
            num_atoms=na,
            num_global_nodes=nv,
            num_edges_aa=2 * nb,
            num_edges_va=na * nv,
        )

        # reference bond to atom map

        bond_to_atom_map = {0: [1, 3], 1: [3, 4], 2: [0, 2], 3: [0, 4]}

        atom_map_number = [list(range(na))]
        if nv == 0:
            global_map_number = [[]]
        elif nv == 1:
            global_map_number = [[0]]
        else:
            raise ValueError

        assert_graph_connectivity(
            g,
            num_global_nodes=nv,
            bond_to_atom_map=bond_to_atom_map,
            atom_map_number=atom_map_number,
            global_map_number=global_map_number,
        )


def assert_graph_quantity(g, num_atoms, num_global_nodes, num_edges_aa, num_edges_va):
    nodes = ["atom"]
    edges = ["bond"]
    ref_num_nodes = [num_atoms]
    ref_num_edges = [num_edges_aa]

    if num_global_nodes > 0:
        nodes.append("global")
        edges.extend(["a2g", "g2a"])
        ref_num_nodes.append(num_global_nodes)
        ref_num_edges.extend([num_edges_va, num_edges_va])

    num_nodes = [g.number_of_nodes(n) for n in nodes]
    num_edges = [g.number_of_edges(e) for e in edges]

    assert set(g.ntypes) == set(nodes)
    assert set(g.etypes) == set(edges)
    assert num_nodes == ref_num_nodes
    assert num_edges == ref_num_edges


def assert_graph_connectivity(
    g, num_global_nodes, bond_to_atom_map, atom_map_number, global_map_number
):

    # test atom to atom connection
    etype = "bond"
    src, dst, eid = g.edges(form="all", order="eid", etype=etype)
    pairs = [(s, d) for s, d in zip(src.numpy().tolist(), dst.numpy().tolist())]
    for b, atoms in bond_to_atom_map.items():
        x = {tuple(atoms), tuple(reversed(atoms))}
        y = {pairs[2 * b], pairs[2 * b + 1]}
        assert x == y

    # NOTE, unnecessary to check global node edges, since no feats will be assigned
    if num_global_nodes > 0:
        # test global to atom connection
        etype = "g2a"
        src, dst, eid = g.edges(form="all", order="eid", etype=etype)
        i = 0
        for vv, aa in zip(global_map_number, atom_map_number):
            for v in vv:
                for a in aa:
                    assert src[i] == v
                    assert dst[i] == a
                    i += 1

        # test atom to global connection
        etype = "a2g"
        src, dst, eid = g.edges(form="all", order="eid", etype=etype)
        i = 0
        for vv, aa in zip(global_map_number, atom_map_number):
            for v in vv:
                for a in aa:
                    assert src[i] == a
                    assert dst[i] == v
                    i += 1


def assert_combine_graphs(
    graphs,
    na,
    nb,
    nv,
    atom_map_number,
    bond_map_number,
    global_map_number,
    bond_to_atom_map,
):

    ne_v = sum(i * j for i, j in zip(na, nv))  # number of atom-global edges
    na = sum(na)
    nb = sum(nb)
    nv = sum(nv)
    ne_a = 2 * nb  # num of atom-atom edges

    g = combine_graphs(graphs, atom_map_number, bond_map_number)

    assert_graph_quantity(g, na, nv, ne_a, ne_v)
    assert_graph_connectivity(
        g, nv, bond_to_atom_map, atom_map_number, global_map_number
    )

    #
    # test features
    #
    feats_atom = torch.cat([g.nodes["atom"].data["feat"] for g in graphs])
    feats_bond = torch.cat([g.edges["bond"].data["feat"] for g in graphs])

    reorder_atom = np.concatenate(atom_map_number).tolist()
    reorder_atom = [reorder_atom.index(i) for i in range(len(reorder_atom))]
    reorder_bond = np.concatenate(bond_map_number).tolist()
    if reorder_bond:
        reorder_bond = np.concatenate(
            [[2 * i, 2 * i + 1] for i in reorder_bond]
        ).tolist()
        reorder_bond = [reorder_bond.index(i) for i in range(len(reorder_bond))]

    assert torch.equal(g.nodes["atom"].data["feat"], feats_atom[reorder_atom])
    assert torch.equal(g.edges["bond"].data["feat"], feats_bond[reorder_bond])

    if nv > 0:
        feats_global = torch.cat([g.nodes["global"].data["feat"] for g in graphs])
        reorder_global = np.concatenate(global_map_number).tolist()
        reorder_global = [reorder_global.index(i) for i in range(len(reorder_global))]

        assert torch.equal(g.nodes["global"].data["feat"], feats_global[reorder_global])
