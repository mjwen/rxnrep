"""
Build dgl graphs from molecules.
"""

import itertools
import logging
from collections import defaultdict
from typing import Callable, List, Optional, Tuple

import dgl
import torch

from rxnrep.core.molecule import Molecule
from rxnrep.core.reaction import Reaction

logger = logging.getLogger(__name__)


def mol_to_graph(mol: Molecule, num_global_nodes: int = 0) -> dgl.DGLGraph:
    """
    Create dgl graph for a molecule.

    Atoms will be nodes (with node type `atom`) of the dgl graph; atom i be node i.
    Bonds will be edges (with edge type `bond`, i.e. atom to atom). The is a bi-directed
    graph and each bond corresponds to two edges: bond j corresponds to edges 2j and
    2j+1.

    if `num_global_node > 0`, the corresponding number of global nodes are created
    (with node type 'global'). Each global node is connected to all the atom nodes via
    edges of type `a2g` and `g2a`).

    Args:
        mol: molecule
        num_global_nodes: number of global nodes to add. e.g. node holding global
            features of molecules. In the literature (e.g. MPNN), global nodes is called
            virtual nodes.

    Returns:
        dgl graph for the molecule
    """

    edges_dict = {}

    # atom to atom nodes
    a2a = []
    for i, j in mol.bonds:
        a2a.extend([[i, j], [j, i]])

    edges_dict[("atom", "bond", "atom")] = a2a
    num_nodes_dict = {"atom": mol.num_atoms}

    # global nodes
    if num_global_nodes > 0:
        a2v = []
        v2a = []
        for a in range(mol.num_atoms):
            for v in range(num_global_nodes):
                a2v.append([a, v])
                v2a.append([v, a])

        edges_dict[("atom", "a2g", "global")] = a2v
        edges_dict[("global", "g2a", "atom")] = v2a
        num_nodes_dict["global"] = num_global_nodes

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
    i.e. `bond` edges according to bond map numbers. The global nodes (if any) is
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

            # deal with nodes (i.e. atom and optionally global)
            if srctype == "atom":
                src = [atom_map_number[i][j] for j in u]
            else:
                # global nodes
                src = u + num_nodes_dict[srctype]
                src = src.numpy().tolist()

            if dsttype == "atom":
                dst = [atom_map_number[i][j] for j in v]
            else:
                # global nodes
                dst = v + num_nodes_dict[dsttype]
                dst = dst.numpy().tolist()

            edges_dict[rel].extend([(s, d) for s, d in zip(src, dst)])

        for ntype in ntypes:
            num_nodes_dict[ntype] += g.number_of_nodes(ntype)

    # reorder bond edges (bond edges)
    bond_map_number_list = []
    for i in itertools.chain.from_iterable(bond_map_number):
        bond_map_number_list.extend([2 * i, 2 * i + 1])
    bond_reorder = [
        bond_map_number_list.index(i) for i in range(len(bond_map_number_list))
    ]

    rel = ("atom", "bond", "atom")
    a2a_edges = edges_dict.pop(rel)
    a2a_edges = [a2a_edges[i] for i in bond_reorder]

    edges_dict[rel] = a2a_edges

    # create graph
    new_g = dgl.heterograph(edges_dict, num_nodes_dict=num_nodes_dict)

    # Batch features

    # reorder node features (atom and global)
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

        if etype == "bond":
            new_feats = {k: v[bond_reorder] for k, v in new_feats.items()}

        new_g.edges[etype].data.update(new_feats)

    # add _ID to atom feature
    new_g.nodes["atom"].data["_ID"] = torch.arange(new_g.num_nodes("atom"))

    return new_g


def create_reaction_graph(
    reactants_graph: dgl.DGLGraph,
    products_graph: dgl.DGLGraph,
    num_unchanged_bonds: int,
    num_lost_bonds: int,
    num_added_bonds: int,
    num_global_nodes: int = 0,
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
    3. we create `num_global_nodes` global nodes and each id bi-direct connected to
       every atom node.

    This assumes the lost bonds in the reactants (or added bonds in the products) have
    larger node number than unchanged bonds. This is the case if
    :meth:`Reaction.get_reactants_bond_map_number()`
    and
    :meth:`Reaction.get_products_bond_map_number()`
    are used to generate the bond map number when `combine_graphs()`.

    This also assumes the reactants_graph and products_graph are created by
    `combine_graphs()`.

    The order of the atom nodes is unchanged. The bonds
    (i.e. `bond` edges between atoms) are reordered. Actually, The bonds in the
    reactants are intact and the added bonds in the products are updated to come after
    the bonds in the reactants.
    More specifically, according to :meth:`Reaction.get_reactants_bond_map_number()`
    and :meth:`Reaction.get_products_bond_map_number()`, bond 0, 1, ..., N_un-1 are the
    unchanged bonds, N_un, ..., N-1 are the lost bonds in the reactants, and N, ...,
    N+N_add-1 are the added bonds in the products, where N_un is the number of unchanged
    bonds, N is the number of bonds in the reactants (i.e. unchanged plus lost), and
    N_add is the number if added bonds (i.e. total number of bonds in the products
    minus unchanged bonds).

    Args:
        reactants_graph: the graph of the reactants, Note this should be the combined
            graph for all molecules in the reactants.
        products_graph: the graph of the reactants, Note this should be the combined
            graph for all molecules in the reactants.
        num_unchanged_bonds: number of unchanged bonds in the reaction.
        num_lost_bonds: number of lost bonds in the reactants.
        num_added_bonds: number of added bonds in the products.
        num_global_nodes: number of global nodes to add. e.g. node holding global
            features of molecules.

    Returns:
        A graph representing the reaction.
    """

    # First add unchanged bonds and lost bonds from reactants
    rel = ("atom", "bond", "atom")
    src, dst = reactants_graph.edges(order="eid", etype=rel)
    a2a = [(u, v) for u, v in zip(src, dst)]

    # Then add added bonds from products
    src, dst, eid = products_graph.edges(form="all", order="eid", etype=rel)
    for u, v, e in zip(src, dst, eid):
        # e // 2 because two edges for each bond
        if e // 2 >= num_unchanged_bonds:
            a2a.append((u, v))

    num_atoms = reactants_graph.num_nodes("atom")
    edges_dict = {("atom", "bond", "atom"): a2a}
    num_nodes_dict = {"atom": num_atoms}

    # global nodes
    if num_global_nodes > 0:
        a2v = []
        v2a = []
        for a in range(num_atoms):
            for v in range(num_global_nodes):
                a2v.append([a, v])
                v2a.append([v, a])

        edges_dict[("atom", "a2g", "global")] = a2v
        edges_dict[("global", "g2a", "atom")] = v2a
        num_nodes_dict["global"] = num_global_nodes

    g = dgl.heterograph(edges_dict, num_nodes_dict=num_nodes_dict)

    return g


def build_graph_and_featurize_reaction(
    reaction: Reaction,
    atom_featurizer: Callable,
    bond_featurizer: Callable,
    global_featurizer: Optional[Callable] = None,
    num_global_nodes: int = 1,
) -> Tuple[dgl.DGLGraph, dgl.DGLGraph, dgl.DGLGraph]:
    """
    Build dgl graphs for the reactants and products in a reaction and featurize them.

    Args:
        reaction:
        atom_featurizer:
        bond_featurizer:
        global_featurizer: If `num_global_nodes > 0`, this featurizer generates
            features for the global nodes.
        num_global_nodes: Number of global nodes to create. Each global will be
            bi-directionally connected to all atom nodes.

    Returns:
        reactants_g: dgl graph for the reactants. One graph for all reactants; each
            disjoint subgraph for a molecule.
        products_g: dgl graph for the products. One graph for all reactants; each
            disjoint subgraph for a molecule.
        reaction_g: dgl graph for the reaction. bond edges is the union of reactants
            bonds and products bonds. See `create_reaction_graph()` for more.
    """

    def featurize_one_mol(m: Molecule):
        g = mol_to_graph(m, num_global_nodes)

        rdkit_mol = m.rdkit_mol

        atom_feats = atom_featurizer(rdkit_mol)
        bond_feats = bond_featurizer(rdkit_mol)
        # each bond corresponds to two edges in the graph
        bond_feats = torch.repeat_interleave(bond_feats, 2, dim=0)

        g.nodes["atom"].data.update({"feat": atom_feats})
        g.edges["bond"].data.update({"feat": bond_feats})

        if num_global_nodes > 0:
            global_feats = global_featurizer(
                rdkit_mol, charge=m.charge, environment=m.environment
            )
            g.nodes["global"].data.update({"feat": global_feats})

        return g

    try:
        reactant_graphs = [featurize_one_mol(m) for m in reaction.reactants]
        product_graphs = [featurize_one_mol(m) for m in reaction.products]

        # combine small graphs to form one big graph for reactants and products
        atom_map_number = reaction.get_reactants_atom_map_number(zero_based=True)
        bond_map_number = reaction.get_reactants_bond_map_number(for_changed=True)
        reactants_g = combine_graphs(reactant_graphs, atom_map_number, bond_map_number)

        atom_map_number = reaction.get_products_atom_map_number(zero_based=True)
        bond_map_number = reaction.get_products_bond_map_number(for_changed=True)
        products_g = combine_graphs(product_graphs, atom_map_number, bond_map_number)

        # combine reaction graph from the combined reactant graph and product graph
        num_unchanged = len(reaction.unchanged_bonds)
        num_lost = len(reaction.lost_bonds)
        num_added = len(reaction.added_bonds)

        reaction_g = create_reaction_graph(
            reactants_g,
            products_g,
            num_unchanged,
            num_lost,
            num_added,
            num_global_nodes,
        )

    except Exception as e:
        logger.error(f"Error build graph and featurize for reaction: {reaction.id}")
        raise Exception(e)

    return reactants_g, products_g, reaction_g
