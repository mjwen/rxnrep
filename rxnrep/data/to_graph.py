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
    num_nodes_dict = {"atom": mol.num_atoms}

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
        num_nodes_dict["virtual"] = num_virtual_nodes

    g = dgl.heterograph(edges_dict, num_nodes_dict=num_nodes_dict)

    return g


def combine_graphs(
    graphs: List[dgl.DGLGraph],
    atom_map_number: List[List[int]],
    bond_map_number: List[List[int]],
) -> dgl.DGLGraph:
    """
    Combine a sequence of dgl graphs and their features to form a new graph.

    For example, combine the graphs of all reactants (or products) in a reaction.

    This is different from batching where the nodes and features are concatenated.
    Here we reorder atom nodes according the atom map number, and reorder bond edges
    i.e. `a2a` edges according to bond map numbers. The virtual nodes (if any) is
    simply batched.

    This assumes the graphs are created with `mol_to_graph()` in this file.

    Args:
        graphs: a sequence of dgl graphs
        atom_map_number: the order of atoms for the combined dgl graph. Each inner list
            gives the order of the atoms for a graph in graphs. Should start from 0.
            `atom_map_numbers` should have non-repetitive values from 0 to N-1, where N
            is the total number of atoms in the graphs.
        bond_map_number: the order of bonds for the combined dgl graph. Each inner list
            gives the order of the bonds for a graph in graphs. Should start from 0.
            `bond_map_numbers` should have non-repetitive values from 0 to N-1, where N
            is the total number of bonds in the graphs.

    Returns:
        dgl graph with atom nodes and bond edges reordered
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
    new_g = dgl.heterograph(edges_dict, num_nodes_dict=num_nodes_dict)

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


def create_reaction_graph(
    reactants_graph: dgl.DGLGraph,
    products_graph: dgl.DGLGraph,
    num_unchanged_bonds: int,
    num_lost_bonds: int,
    num_added_bonds: int,
    self_loop: bool = False,
) -> dgl.DGLGraph:
    """
    Create a reaction graph from the reactants graph and the products graph.

    It is expected to take the difference of features between the products and the
    reactants.

    The created graph has the below characteristics:
    1. has the same number of atom nodes as in the reactants and products;
    2. the bonds (i.e. atom to atom edges) are the union of the bonds in the reactants
       and the products. i.e. unchanged bonds, lost bonds in reactants, and added bonds
       in products;
    3. a single global virtual nodes if virtual nodes is present in the input graphs.
       If the reactants (products) have multiple molecules, there will be multiple
       global virtual nodes for them; then it is expected to aggregate the
       global features of the reactants (products) and then use this graph.

    This assumes the lost bonds in the reactants (or added bonds in the products) have
    larger node number than unchanged bonds. This is the case if
    :meth:`Reaction.get_reactants_bond_map_number()`
    and
    :meth:`Reaction.get_products_bond_map_number()`
    are used to generate the bond map number when `combine_graphs()`.

    This also assumes the reactants_graph and products_graph are created by
    `combine_graphs()`.

    The order of the atom nodes (and virtual nodes if present) is unchanged. The bonds
    (i.e. `a2a` edges between atoms) are reordered. Actually, The bonds in the
    reactants are intact and the added bonds in the products are updated to come after
    the bonds in the reactants.
    More specifically, bond nodes 0, 1, ..., N_un-1 are the unchanged bonds,
    N_un, ..., N-1 are the lost bonds, and N, ..., N+N_add-1 are the added bonds,
    where N_un is the number of unchanged bonds, N is the number of bonds in the
    reactants (i.e. unchanged plus lost), and N_add is the number if added bonds.

    Args:
        reactants_graph: the graph of the reactants, Note this should be the combined
            graph for all molecules in the reactants.
        products_graph: the graph of the reactants, Note this should be the combined
            graph for all molecules in the reactants.
        num_unchanged_bonds: number of unchanged bonds in the reaction.
        num_lost_bonds: number of lost bonds in the reactants.
        num_added_bonds: number of added bonds in the products.
        self_loop: whether to add self loop for each node.

    Returns:
        A graph representing the reaction.
    """

    # Construct edges between atoms and bonds

    # Let bonds 0, 1, ..., N_un-1 be unchanged bonds, N_un, ..., N-1 be lost bonds, and
    # N, ..., N+N_add-1 be the added bonds, where N_un is the number of unchanged bonds,
    # N is the number of bonds in the reactants (i.e. unchanged plus lost), and N_add
    # is the number if added bonds.

    # first add unchanged bonds and lost bonds from reactants
    rel = ("atom", "a2b", "bond")
    src, dst = reactants_graph.edges(order="eid", etype=rel)
    a2b = [(u, v) for u, v in zip(src, dst)]

    rel = ("bond", "b2a", "atom")
    src, dst = reactants_graph.edges(order="eid", etype=rel)
    b2a = [(u, v) for u, v in zip(src, dst)]

    # then add added bonds
    rel = ("atom", "a2b", "bond")
    src, dst = products_graph.edges(order="eid", etype=rel)
    for u, v in zip(src, dst):
        if v >= num_unchanged_bonds:  # select added bonds
            # NOTE, should not v += num_lost_bonds. doing this will alter dst.
            v = v + num_lost_bonds  # shift bond nodes to be after lost bonds
            a2b.append((u, v))

    rel = ("bond", "b2a", "atom")
    src, dst = products_graph.edges(order="eid", etype=rel)
    for u, v in zip(src, dst):
        if u >= num_unchanged_bonds:  # select added bonds
            # NOTE, should not u += num_lost_bonds. doing this will alter src.
            u = u + num_lost_bonds  # shift bond nodes to be after lost bonds
            b2a.append((u, v))

    # Construct edges between global and atoms (bonds)

    num_atoms = reactants_graph.num_nodes("atom")
    num_bonds = num_unchanged_bonds + num_lost_bonds + num_added_bonds

    a2g = [(a, 0) for a in range(num_atoms)]
    g2a = [(0, a) for a in range(num_atoms)]
    b2g = [(b, 0) for b in range(num_bonds)]
    g2b = [(0, b) for b in range(num_bonds)]

    edges_dict = {
        ("atom", "a2b", "bond"): a2b,
        ("bond", "b2a", "atom"): b2a,
        ("atom", "a2g", "global"): a2g,
        ("global", "g2a", "atom"): g2a,
        ("bond", "b2g", "global"): b2g,
        ("global", "g2b", "bond"): g2b,
    }

    if self_loop:
        a2a = [(i, i) for i in range(num_atoms)]
        b2b = [(i, i) for i in range(num_bonds)]
        g2g = [(0, 0)]
        edges_dict.update(
            {
                ("atom", "a2a", "atom"): a2a,
                ("bond", "b2b", "bond"): b2b,
                ("global", "g2g", "global"): g2g,
            }
        )

    g = dgl.heterograph(edges_dict)

    return g
