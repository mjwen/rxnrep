import dgl
import numpy as np
import torch

from rxnrep.core.molecule import Molecule
from rxnrep.data.to_graph import combine_graphs, mol_to_graph


def create_graph(m, n_a, n_b, n_v, self_loop):
    """

    Args:
        m: molecule
        n_a:  number of atoms
        n_b:  number of bonds
        n_v:  number of virtual nodes
    """

    g = mol_to_graph(m, num_virtual_nodes=n_v, self_loop=self_loop)

    g.nodes["atom"].data.update({"feat": torch.arange(n_a * 4).reshape(n_a, 4)})

    # this create zero tensor (i.e. of shape (0, 3)) if n_b == 0
    bond_feats = torch.arange(n_b * 3).reshape(n_b, 3)
    bond_feats = torch.repeat_interleave(
        bond_feats, torch.tensor([2] * n_b).long(), dim=0
    )
    g.edges["a2a"].data.update({"feat": bond_feats})

    if n_v > 0:
        g.nodes["virtual"].data.update({"feat": torch.arange(n_v * 2).reshape(n_v, 2)})

    return g


def create_graph_C(num_virtual_nodes, self_loop=False):
    """
    Create a single atom molecule C.

    atom_feats:
        [[0,1,2,3]]

    bond_feats:
        None

    virtual_feats:
        [[0,1],
         [2,3],
         ...]
    """
    smi = "[C]"
    m = Molecule.from_smiles(smi)

    return create_graph(m, 1, 0, num_virtual_nodes, self_loop)


def create_graph_CO2(num_virtual_nodes, self_loop=False):
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
         [0,1,2],
         [3,4,5],
         [3,4,5]]
    Note [0,1,2] corresponds to two edges for the same bond.


    global_feat:
        [[0,1],
         [2,3],
         ... ]

    """
    smi = "O=C=O"
    m = Molecule.from_smiles(smi)

    return create_graph(m, 3, 2, num_virtual_nodes, self_loop)


def test_create_graph():
    def assert_one(g, n_a, n_b, n_v):

        num_nodes = {"atom": n_a}
        num_edges = {"a2a": 2 * n_b}

        if n_v > 0:
            num_nodes["virtual"] = n_v
            num_edges["v2a"] = n_a * n_v
            num_edges["a2v"] = n_a * n_v

        assert set(g.ntypes) == set(num_nodes.keys())
        assert set(g.etypes) == set(num_edges.keys())
        for k, n in num_nodes.items():
            assert g.number_of_nodes(k) == n
        for k, n in num_edges.items():
            assert g.number_of_edges(k) == n

    for n_v in range(3):
        g = create_graph_C(num_virtual_nodes=n_v)
        assert_one(g, 1, 0, n_v)

    for n_v in range(3):
        g = create_graph_CO2(num_virtual_nodes=n_v)
        assert_one(g, 3, 2, n_v)


def test_batch_graph():

    n_a_1 = 1
    n_b_1 = 0
    n_v_1 = 1
    g1 = create_graph_C(num_virtual_nodes=n_v_1)

    n_a_2 = 3
    n_b_2 = 2
    n_v_2 = 1
    g2 = create_graph_CO2(num_virtual_nodes=n_v_2)

    g = dgl.batch([g1, g2])

    num_nodes = {"atom": n_a_1 + n_a_2, "virtual": n_v_1 + n_v_2}
    num_edges = {
        "a2a": 2 * (n_b_1 + n_b_2),
        "v2a": n_a_1 * n_v_1 + n_a_2 * n_v_2,
        "a2v": n_a_1 * n_v_1 + n_a_2 * n_v_2,
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

    for t in ["a2a"]:
        assert torch.equal(
            g.edges[t].data["feat"],
            torch.cat([g1.edges[t].data["feat"], g2.edges[t].data["feat"]]),
        )


def assert_combine_graphs(
    graphs,
    na,
    nb,
    nv,
    atom_map_number,
    bond_map_number,
    virtual_map_number,
    bond_to_atom_map,
):

    ne_v = sum(i * j for i, j in zip(na, nv))  # number of atom-virtual edges
    na = sum(na)
    nb = sum(nb)
    nv = sum(nv)
    ne_a = 2 * nb  # num of atom-atom edges

    g = combine_graphs(graphs, atom_map_number, bond_map_number)

    nodes = ["atom"]
    edges = ["a2a"]
    ref_num_nodes = [na]
    ref_num_edges = [ne_a]

    if nv > 0:
        nodes.append("virtual")
        edges.extend(["a2v", "v2a"])
        ref_num_nodes.append(nv)
        ref_num_edges.extend([ne_v, ne_v])

    num_nodes = [g.number_of_nodes(n) for n in nodes]
    num_edges = [g.number_of_edges(e) for e in edges]

    assert set(g.ntypes) == set(nodes)
    assert set(g.etypes) == set(edges)
    assert num_nodes == ref_num_nodes
    assert num_edges == ref_num_edges

    #
    # test structure
    #

    # test atom to atom connection
    etype = "a2a"
    src, dst, eid = g.edges(form="all", order="eid", etype=etype)
    pairs = [(s, d) for s, d in zip(src.numpy().tolist(), dst.numpy().tolist())]
    for b, atoms in bond_to_atom_map.items():
        x = {tuple(atoms), tuple(reversed(atoms))}
        y = {pairs[2 * b], pairs[2 * b + 1]}
        assert x == y

    if nv > 0:
        # test virtual to atom connection
        etype = "v2a"
        src, dst, eid = g.edges(form="all", order="eid", etype=etype)
        i = 0
        for vv, aa in zip(virtual_map_number, atom_map_number):
            for v in vv:
                for a in aa:
                    assert src[i] == v
                    assert dst[i] == a
                    i += 1

        # test atom to virtual connection
        etype = "a2v"
        src, dst, eid = g.edges(form="all", order="eid", etype=etype)
        i = 0
        for vv, aa in zip(virtual_map_number, atom_map_number):
            for v in vv:
                for a in aa:
                    assert src[i] == a
                    assert dst[i] == v
                    i += 1

    #
    # test features
    #

    feats_atom = torch.cat([g.nodes["atom"].data["feat"] for g in graphs])
    feats_bond = torch.cat([g.edges["a2a"].data["feat"] for g in graphs])

    reorder_atom = np.concatenate(atom_map_number).tolist()
    reorder_atom = [reorder_atom.index(i) for i in range(len(reorder_atom))]
    reorder_bond = np.concatenate(bond_map_number).tolist()
    if reorder_bond:
        reorder_bond = np.concatenate(
            [[2 * i, 2 * i + 1] for i in reorder_bond]
        ).tolist()
        reorder_bond = [reorder_bond.index(i) for i in range(len(reorder_bond))]

    assert torch.equal(g.nodes["atom"].data["feat"], feats_atom[reorder_atom])
    assert torch.equal(g.edges["a2a"].data["feat"], feats_bond[reorder_bond])

    if nv > 0:
        feats_virtual = torch.cat([g.nodes["virtual"].data["feat"] for g in graphs])
        reorder_virtual = np.concatenate(virtual_map_number).tolist()
        reorder_virtual = [
            reorder_virtual.index(i) for i in range(len(reorder_virtual))
        ]

        assert torch.equal(
            g.nodes["virtual"].data["feat"], feats_virtual[reorder_virtual]
        )


def test_combine_graphs_CO2():
    """
    Three CO2 mols.
    """

    for nv in [0, 1]:
        g = create_graph_CO2(num_virtual_nodes=nv)

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
            virtual_map_number = [[], [], []]
        elif nv == 1:
            virtual_map_number = [[0], [1], [2]]
        else:
            raise ValueError

        assert_combine_graphs(
            [g, g, g],
            [3, 3, 3],
            [2, 2, 2],
            [nv, nv, nv],
            atom_map_number,
            bond_map_number,
            virtual_map_number,
            bond_to_atom_map,
        )


def test_combine_graphs_C():
    """
    Three single atom molecule without bond.
    """

    for nv in [0, 1]:
        g = create_graph_C(num_virtual_nodes=nv)

        atom_map_number = [[1], [2], [0]]
        bond_map_number = [[], [], []]
        bond_to_atom_map = {}

        if nv == 0:
            virtual_map_number = [[], [], []]
        elif nv == 1:
            virtual_map_number = [[0], [1], [2]]
        else:
            raise ValueError

        assert_combine_graphs(
            [g, g, g],
            [1, 1, 1],
            [0, 0, 0],
            [nv, nv, nv],
            atom_map_number,
            bond_map_number,
            virtual_map_number,
            bond_to_atom_map,
        )


def test_combine_graphs_C_CO2():
    """
    CO2, C and CO2; and C, CO2 and C.
    """

    for nv in [0, 1]:
        g1 = create_graph_CO2(num_virtual_nodes=nv)
        g2 = create_graph_C(num_virtual_nodes=nv)

        if nv == 0:
            virtual_map_number = [[], [], []]
        elif nv == 1:
            virtual_map_number = [[0], [1], [2]]
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
            virtual_map_number,
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
            virtual_map_number,
            bond_to_atom_map,
        )
