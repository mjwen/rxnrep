"""
Build molecule graphs.
"""
import itertools
from collections import defaultdict
import dgl
import torch
import numpy as np
from rdkit import Chem
from typing import List


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

    # bonds
    num_bonds = mol.GetNumBonds()

    # If no bonds (e.g. H+), create an artifact bond and connect it to the 1st atom
    if num_bonds == 0:
        num_bonds = 1
        a2b = [(0, 0)]
        b2a = [(0, 0)]
    else:
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

    if num_bonds == 0:
        num_bonds = 1
        a2b = [(0, 0)]
        b2a = [(0, 0)]
    else:
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
    graphs: List[dgl.DGLGraph], atom_map_number: List[List[int]]
) -> dgl.DGLGraph:
    """
    Combine a sequence of dgl graphs and their features to form a new graph.

    This is different from batching where the nodes and features are concatenated.
    Here we reorder atom nodes according the atom map number.

    Args:
        graphs: a sequence of dgl graphs
        atom_map_number: the order of atoms for the combined dgl graph. Each inner list
            gives the order of the atoms for a graph in graphs.

    Returns:
        dgl graph
    """

    if len(graphs) == 1:
        return graphs[0]

    atom_map_number = convert_atom_map_number(graphs, atom_map_number)

    relations = graphs[0].canonical_etypes
    ntypes = graphs[0].ntypes

    # Batch graph structure for each relation graph
    edge_dict = defaultdict(list)
    num_nodes_dict = defaultdict(int)

    for i, g in enumerate(graphs):
        for rel in relations:
            srctype, etype, dsttype = rel
            u, v = g.edges(order="eid", etype=rel)

            if srctype == "atom":
                # convert atom map number (do not need to add u since atom map is global)
                src = torch.tensor([atom_map_number[i][j] for j in u])
            else:
                src = u + num_nodes_dict[srctype]

            if dsttype == "atom":
                # convert atom map number (do not need to add u since atom map is global)
                dst = torch.tensor([atom_map_number[i][j] for j in v])
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
    reorder = [atom_map_number_list.index(i) for i in range(len(atom_map_number_list))]

    for ntype in graphs[0].ntypes:
        feat_dicts = [g.nodes[ntype].data for g in graphs if g.number_of_nodes(ntype) > 0]

        if len(feat_dicts) == 0:
            new_feats = {}
        else:
            # concatenate features
            keys = feat_dicts[0].keys()
            new_feats = {k: torch.cat([fd[k] for fd in feat_dicts], 0) for k in keys}

            # reorder atom features according to atom_map_number
            if ntype == "atom":
                new_feats = {k: v[reorder] for k, v in new_feats.items()}

        new_g.nodes[ntype].data.update(new_feats)

    return new_g


def convert_atom_map_number(
    graphs: List[dgl.DGLGraph], atom_map_number: List[List[int]],
) -> List[List[int]]:
    """
    Convert the atom map number from 1 based to 0 based.

    Before that, check the correctness the atom map number to ensure that
    (1) it has the same size as the total number of atom nodes in the graphs and
    (2) the map numbers are unique.

    Args:
        graphs: a sequence of dgl graphs
        atom_map_number: the order of atoms for the combined dgl graph. Each inner list
            gives the order of the atoms for a graph in graphs.

    Returns:
        converted_atom_map_number: same as the input atom_map_number
    """
    num_nodes = sum([g.num_nodes("atom") for g in graphs])

    # convert to 0 based
    new_atom_map_number = [list(np.asarray(x) - 1) for x in atom_map_number]

    map_num_1d = [x for x in itertools.chain.from_iterable(new_atom_map_number)]
    if set(map_num_1d) != set(range(num_nodes)):
        raise ValueError(
            f"Incorrect atom map numbers. Expect unique ones, but got {atom_map_number}."
        )

    return new_atom_map_number
