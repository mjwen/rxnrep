"""
Build molecule graphs.
"""
import itertools
from collections import defaultdict
import dgl
import torch
from rdkit import Chem
from typing import List, Tuple


def create_hetero_molecule_graph(mol: Chem.Mol, self_loop: bool = False) -> dgl.DGLGraph:
    """
    Create a heterogeneous molecule graph for an rdkit molecule.

    The created graph represent each atom and each bond as a node. The structure of the
    molecule is preserved: an atom node is connected to all bond nodes the atom taking
    part in forming the bond. Similarly, an bond node is connected to the two atom
    nodes forming the nodes. We also create a global node that is connected to all
    atom and bond nodes.

    Atom i corresponds to graph node (with type `atom`) i.
    Bond i corresponds to graph node (with type `bond`) i.
    There is only one global state node 0 (with type `global`).

    Args:
        mol: rdkit molecule
        self_loop: whether to create self loop for the nodes, i.e. add an edge for each
            node connecting to it self.

    Returns:
        A dgl graph
    """
    num_atoms = mol.GetNumAtoms()
    num_bonds = mol.GetNumBonds()

    a2b = []
    b2a = []
    for b in range(num_bonds):
        bond = mol.GetBondWithIdx(b)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        b2a.extend([[b, u], [b, v]])
        a2b.extend([[u, b], [v, b]])

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


def create_hetero_complete_graph(mol: Chem.Mol, self_loop: bool = False) -> dgl.DGLGraph:
    """
    Create a complete graph from rdkit molecule, a complete graph where all atoms are
    connected to each other.

    Atom, bonds, and global states are all represented as nodes in the graph.
    Atom i corresponds to graph node (with type `atom`) i.
    There is only one global state node 0 (with type `global`).

    Bonds is different from the typical notion. Here we assume there is a bond between
    every atom pairs.

    The order of the bonds are (0,1), (0,2), ... , (0, N-1), (1,2), (1,3), ...,
    (N-2, N-1), where N is the number of atoms.

    Args:
        mol: rdkit molecule
        self_loop: whether to create self loop for the nodes, i.e. add an edge for each
            node connecting to it self.

    Returns:
        A dgl graph
    """

    num_atoms = mol.GetNumAtoms()
    num_bonds = num_atoms * (num_atoms - 1) // 2

    a2b = []
    b2a = []
    for b, (u, v) in enumerate(itertools.combinations(range(num_atoms), 2)):
        b2a.extend([[b, u], [b, v]])
        a2b.extend([[u, b], [v, b]])

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


def combine_graphs(
    graphs: List[dgl.DGLGraph],
    atom_map_number: List[List[int]],
    bond_map_number: List[List[int]],
) -> dgl.DGLGraph:
    """
    Combine a sequence of dgl graphs and their features to form a new graph.

    This is different from batching where the nodes and features are concatenated.
    Here we reorder atom nodes according the atom map number, and reorder bond nodes
    according to bond map numbers.

    Args:
        graphs: a sequence of dgl graphs
        atom_map_number: the order of atoms for the combined dgl graph. Each inner list
            gives the order of the atoms for a graph in graphs. Should start from 0.
        bond_map_number: the order of bonds for the combined dgl graph. Each inner list
            gives the order of the bonds for a graph in graphs. Should start from 0.
            A value of `None` means the bond is not mapped between the reactants and
            the products.

    Returns:
        dgl graph
    """

    # create bond node reorder map number for unchanged, changed, and artificial bonds
    bond_map_number = get_bond_node_reorder_map_number(bond_map_number)

    # Batch graph structure for each relation graph

    relations = graphs[0].canonical_etypes
    ntypes = graphs[0].ntypes

    edge_dict = defaultdict(list)
    num_nodes_dict = defaultdict(int)

    for i, g in enumerate(graphs):
        for rel in relations:
            srctype, etype, dsttype = rel
            u, v = g.edges(order="eid", etype=rel)

            if srctype == "atom":
                src = torch.tensor([atom_map_number[i][j] for j in u])
            elif srctype == "bond":
                src = torch.tensor([bond_map_number[i][j] for j in u])
            else:
                src = u + num_nodes_dict[srctype]

            if dsttype == "atom":
                dst = torch.tensor([atom_map_number[i][j] for j in v])
            elif dsttype == "bond":
                dst = torch.tensor([bond_map_number[i][j] for j in v])
            else:
                dst = v + num_nodes_dict[dsttype]

            edge_dict[rel].append((src, dst))

        for ntype in ntypes:
            num_nodes_dict[ntype] += g.number_of_nodes(ntype)

    for rel in relations:
        src, dst = zip(*edge_dict[rel])
        edge_dict[rel] = (torch.cat(src), torch.cat(dst))

    new_g = dgl.heterograph(edge_dict)

    # Batch node feature

    # prepare for reordering atom features
    atom_map_number_list = list(itertools.chain.from_iterable(atom_map_number))
    bond_map_number_list = list(itertools.chain.from_iterable(bond_map_number))
    atom_reorder = [
        atom_map_number_list.index(i) for i in range(len(atom_map_number_list))
    ]
    bond_reorder = [
        bond_map_number_list.index(i) for i in range(len(bond_map_number_list))
    ]

    for ntype in graphs[0].ntypes:
        feat_dicts = [g.nodes[ntype].data for g in graphs if g.number_of_nodes(ntype) > 0]

        if len(feat_dicts) == 0:
            new_feats = {}
        else:
            # concatenate features
            keys = feat_dicts[0].keys()
            new_feats = {k: torch.cat([fd[k] for fd in feat_dicts], 0) for k in keys}

            # reorder atom features
            if ntype == "atom":
                new_feats = {k: v[atom_reorder] for k, v in new_feats.items()}

            # reorder bond features
            elif ntype == "bond":
                new_feats = {k: v[bond_reorder] for k, v in new_feats.items()}

        new_g.nodes[ntype].data.update(new_feats)

    return new_g


def get_bond_node_reorder_map_number(bond_map_number: List[List[int]]) -> List[List[int]]:
    """
    Get the reorder map number for bond nodes.

    In a graph, there are two possible categories of bond nodes:

    1. unchanged bond nodes: bonds exist in both the reactants and the products.
    2. changed bond nodes: lost bonds in reactants or added bonds in products.
       not exist in both the reactants and products.

    In `bond_map_number`, we only have map number for unchanged bonds (`None` for changed
    bonds). This function creates reorder map number for all bonds. Specifically,

    1. unchanged bonds nodes have map number from 0 to N_un-1;
    2. changed bonds have map number from N_un to N-1;

    Args:
        bond_map_number: the bond map number for all molecules in the reactants or
            products; each inner list is for a molecule.

    Returns:
        reorder_map_number: bond node reorder map number. similar to `bond_map_number`,
            but with map numbers for changed bonds.
    """
    num_unchanged, num_changed = get_num_bond_nodes_information(bond_map_number)

    # (starting) node index for changed bonds
    changed_index = num_unchanged

    reorder_map_number = []
    for number in bond_map_number:
        reorder = []
        for n in number:
            if n is None:
                # changed bond
                reorder.append(changed_index)
                changed_index += 1
            else:
                # unchanged bond
                reorder.append(n)
        reorder_map_number.append(reorder)

    return reorder_map_number


def get_num_bond_nodes_information(bond_map_number: List[List[int]],) -> Tuple[int, int]:
    """
    Get information of the bond nodes.

    In a graph, there are two possible categories of bond nodes:

    1. unchanged bond nodes: bonds exist in both the reactants and the products.
    2. changed bond nodes: lost bonds in reactants or added bonds in products.
       Not exist in both the reactants and products.

    Args:
        bond_map_number: the bond map number for all molecules in the reactants or
            products; each inner list is for a molecule.

    Returns:
        num_unchanged_bond_nodes:
        num_changed_bond_nodes:
    """

    bond_map_number_list = list(itertools.chain.from_iterable(bond_map_number))
    num_unchanged_bond_nodes = len([x for x in bond_map_number_list if x is not None])
    num_changed_bond_nodes = len(bond_map_number_list) - num_unchanged_bond_nodes

    return num_unchanged_bond_nodes, num_changed_bond_nodes
