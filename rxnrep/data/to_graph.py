"""
Build dgl graphs from molecules.
"""

import itertools
from collections import defaultdict
from typing import List

import dgl
import torch

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


def combine_graphs(
    graphs: List[dgl.DGLGraph],
    atom_map_number: List[List[int]],
    bond_map_number: List[List[int]],
) -> dgl.DGLGraph:
    """
    Combine a sequence of dgl graphs and their features to form a new graph.

    This is different from batching where the nodes and features are concatenated.
    Here we reorder atom nodes according the atom map number, and reorder bond edges
    i.e. `a2a` edges according to bond map numbers. The virtual nodes (if any) is
    simply batched.

    This assumes the graphs are created with `mol_to_graph()` in this file.

    Args:
        graphs: a sequence of dgl graphs
        atom_map_number: the order of atoms for the combined dgl graph. Each inner list
            gives the order of the atoms for a graph in graphs. Should start from 0.
        bond_map_number: the order of bonds for the combined dgl graph. Each inner list
            gives the order of the bonds for a graph in graphs. Should start from 0.

    Returns:
        dgl graph
    """

    # Batch graph structure for each relation graph

    relations = graphs[0].canonical_etypes
    ntypes = graphs[0].ntypes

    edges_dict = defaultdict(list)
    num_nodes_dict = defaultdict(int)

    # reorder atom nodes
    for i, g in enumerate(graphs):
        for rel in relations:
            srctype, etype, dsttype = rel
            u, v, eid = g.edges(form="all", order="eid", etype=rel)

            # deal with nodes (i.e. atom and optionally virtual)
            if srctype == "atom":
                src = [atom_map_number[i][j] for j in u]
            else:
                # virtual nodes
                src = u + num_nodes_dict[srctype]
                src = src.numpy().tolist()

            if dsttype == "atom":
                dst = [atom_map_number[i][j] for j in v]
            else:
                # virtual nodes
                dst = v + num_nodes_dict[dsttype]
                dst = dst.numpy().tolist()

            edges_dict[rel].extend([(s, d) for s, d in zip(src, dst)])

        for ntype in ntypes:
            num_nodes_dict[ntype] += g.number_of_nodes(ntype)

    # reorder bond edges (a2a edges)
    bond_map_number_list = []
    for i in itertools.chain.from_iterable(bond_map_number):
        bond_map_number_list.extend([2 * i, 2 * i + 1])
    bond_reorder = [
        bond_map_number_list.index(i) for i in range(len(bond_map_number_list))
    ]

    rel = ("atom", "a2a", "atom")
    a2a_edges = edges_dict.pop(rel)
    a2a_edges = [a2a_edges[i] for i in bond_reorder]

    edges_dict[rel] = list(a2a_edges)

    # create graph
    new_g = dgl.heterograph(edges_dict)

    # Batch features

    # reorder node features (atom and virtual)
    atom_map_number_list = list(itertools.chain.from_iterable(atom_map_number))
    atom_reorder = [
        atom_map_number_list.index(i) for i in range(len(atom_map_number_list))
    ]

    for ntype in graphs[0].ntypes:
        feat_dicts = [g.nodes[ntype].data for g in graphs]

        # concatenate features
        keys = feat_dicts[0].keys()
        new_feats = {k: torch.cat([fd[k] for fd in feat_dicts], 0) for k in keys}

        # reorder atom features
        if ntype == "atom":
            new_feats = {k: v[atom_reorder] for k, v in new_feats.items()}

        new_g.nodes[ntype].data.update(new_feats)

    # reorder edge features (bond)

    for etype in graphs[0].etypes:
        feat_dicts = [g.edges[etype].data for g in graphs]

        # concatenate features
        keys = feat_dicts[0].keys()
        new_feats = {k: torch.cat([fd[k] for fd in feat_dicts], 0) for k in keys}

        if etype == "a2a":
            new_feats = {k: v[bond_reorder] for k, v in new_feats.items()}

        new_g.edges[etype].data.update(new_feats)

    return new_g
