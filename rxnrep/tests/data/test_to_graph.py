import dgl
import torch

from rxnrep.core.molecule import Molecule
from rxnrep.data.to_graph import mol_to_graph


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

    for n_v in [1]:
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
