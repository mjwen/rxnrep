"""
Build molecule graphs.
"""

import dgl

from rxnrep.core.molecule import Molecule


def mol_to_graph(
    mol: Molecule, num_virtual_nodes: int = 0, self_loop: bool = False
) -> dgl.DGLGraph:
    """
    Create dgl graph for a molecule.

    Atoms will be nodes (with node type `atom`) of the dgl graph; atom i be node i.
    Bonds will be edges (with edge type `a2a`, i.e. atom to atom). The is a bi-directed
    graph and each bond corresponds to two edges: bond j corresponds to edges 2j and
    2j+1.

    if `num_virtual_node > 0`, the corresponding number of virtual nodes
    are created (with node type 'virtual'). Each virtual node is connected to all
    the atom nodes via virtual edges (with edge type `a2v` and `v2a`).

    Args:
        mol: molecule
        num_virtual_nodes: number of virtual nodes to add. e.g. node holding global
            features of molecules.
        self_loop: whether to create self loop for atom nodes, i.e. add an edge for each
            atom node connecting to it self.

    Returns:
        dgl graph for the molecule
    """

    edges_dict = {}

    # atom to atom nodes
    a2a = []
    for i, j in mol.bonds:
        a2a.extend([[i, j], [j, i]])

    # atom self loop nodes
    if self_loop:
        a2a.extend([[i, i] for i in range(mol.num_atoms)])

    edges_dict[("atom", "a2a", "atom")] = a2a
    num_nodes = {"atom": mol.num_atoms}

    # virtual nodes
    if num_virtual_nodes > 0:
        a2v = []
        v2a = []
        for a in range(mol.num_atoms):
            for v in range(num_virtual_nodes):
                a2v.append([a, v])
                v2a.append([v, a])

        edges_dict[("atom", "a2v", "virtual")] = a2v
        edges_dict[("virtual", "v2a", "atom")] = v2a
        num_nodes["virtual"] = num_virtual_nodes

    g = dgl.heterograph(edges_dict, num_nodes_dict=num_nodes)

    return g
