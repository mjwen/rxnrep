"""
Build molecule graphs.
"""
import copy
import itertools
from collections import defaultdict
from typing import List, Optional

import dgl
import networkx as nx
import torch
from rdkit import Chem

from rxnrep.core.reaction import Reaction


def create_hetero_molecule_graph(
    mol: Chem.Mol, self_loop: bool = False
) -> dgl.DGLGraph:
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


def create_hetero_complete_graph(
    mol: Chem.Mol, self_loop: bool = False
) -> dgl.DGLGraph:
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

    # Batch graph structure for each relation graph

    relations = graphs[0].canonical_etypes
    ntypes = graphs[0].ntypes

    edges_dict = defaultdict(list)
    num_nodes_dict = defaultdict(int)

    for i, g in enumerate(graphs):
        for rel in relations:
            srctype, etype, dsttype = rel
            u, v = g.edges(order="eid", etype=rel)

            if srctype == "atom":
                src = [atom_map_number[i][j] for j in u]
            elif srctype == "bond":
                src = [bond_map_number[i][j] for j in u]
            else:
                # global nodes
                src = u + num_nodes_dict[srctype]
                src = list(src.numpy())

            if dsttype == "atom":
                dst = [atom_map_number[i][j] for j in v]
            elif dsttype == "bond":
                dst = [bond_map_number[i][j] for j in v]
            else:
                dst = v + num_nodes_dict[dsttype]
                dst = list(dst.numpy())

            edges_dict[rel].extend([(s, d) for s, d in zip(src, dst)])

        for ntype in ntypes:
            num_nodes_dict[ntype] += g.number_of_nodes(ntype)

    new_g = dgl.heterograph(edges_dict)

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
        feat_dicts = [g.nodes[ntype].data for g in graphs]

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

    The created graph has the below characteristics:
    1. has the same number of atom nodes as in reactants and products.
    2. the bond nodes is the union of that of the reactants and the products,
       i.e. unchanged bonds, lost bonds in reactants, and added bonds in products.
    3. a single global nodes.

    This assumes the lost bonds in the reactants (or added bonds in the products) have
    larger node number than unchanged bonds. This is the case if
    :meth:`Reaction.get_reactants_bond_map_number()`
    and
    :meth:`Reaction.get_products_bond_map_number()`
    are used to generate the bond map number when `combine_graphs()`.

    The connection (edges) between atom and bond nodes are preserved. In short,
    the added bonds in the products are appended to the all the bonds in the reactants.
    More specifically, bond nodes 0, 1, ..., N_un-1 are the unchanged bonds,
    N_un, ..., N-1 are the lost bonds, and N, ..., N+N_add-1 are the added bonds,
    where N_un is the number of unchanged bonds, N is the number of bonds in the
    reactants (i.e. unchanged plus lost), and N_add is the number if added bonds.

    The global nodes is connected to every atom and bond node.

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


def get_atom_distance_to_reaction_center(
    reaction: Reaction, max_hop: int = 3
) -> List[int]:
    """
    Get the atom hop distance (graph distance) from the reaction center.

    Reaction center is defined as the atoms involved in broken and added bonds.
    This is done by combining the reactants and products into a single graph,
    retaining all bonds (unchanged, lost, and added).

    Atoms in broken bond has hop distance 0;
    Atoms connected to reaction center via 1 bond has a hop distance of `1`;
    ...
    Atoms connected to reaction center via max_hop or more bonds has a hop distance
    of `max_hop`;
    Atoms in added bond has a hop distance of `max_hop+1`;
    It is possible an atom is in both broken bond and added bond. In such cases,
    we assign it a hop distance of `max_hop+2`.


    Note that, in fact, atoms in added bonds should also have a hop distance of 0.
    But to distinguish such atom with atoms in broken bonds, we give it distance
    `max_hop+1`. We will use the hop distance as a label for training the model,
    so it does not matter whether it is the real distance or not as long as we can
    distinguish them.

    Args:
        reaction: reaction object
        max_hop: maximum number of hops allowed for atoms in unchanged bonds. Atoms
            farther away than this will all have the same distance number.

    Returns:
        Hop distances of the atoms. A list of size N (number of atoms in the reaction)
        and element i is the hop distance for atom i.

    """
    atoms_in_lost_bonds = set(
        [i for i in itertools.chain.from_iterable(reaction.lost_bonds)]
    )
    atoms_in_added_bonds = set(
        [i for i in itertools.chain.from_iterable(reaction.added_bonds)]
    )
    atoms_in_reaction_center = atoms_in_lost_bonds.union(atoms_in_added_bonds)
    all_bonds = reaction.unchanged_bonds + reaction.lost_bonds + reaction.added_bonds

    # distance from center atoms to other atoms
    nx_graph = nx.Graph(incoming_graph_data=all_bonds)
    center_to_others_distance = {}
    for center_atom in atoms_in_reaction_center:
        distances = nx.single_source_shortest_path_length(nx_graph, center_atom)
        center_to_others_distance[center_atom] = distances

    # Atom nodes are ordered according to atom map number in `combine_graphs()`.
    # Here, the atoms in the bonds are also atom map number. So we can directly use then,
    # and the hop_distances will have the same order as the reaction graph,
    # i.e. hop_distances[i] will be the hop distance for atom node i in the reaction
    # graph.
    hop_distances = []
    num_atoms = sum([m.num_atoms for m in reaction.reactants])
    for atom in range(num_atoms):

        # atoms involved with both lost and added bonds
        if atom in atoms_in_lost_bonds and atom in atoms_in_added_bonds:
            hop_distances.append(max_hop + 2)

        # atoms involved only in lost bonds
        elif atom in atoms_in_lost_bonds:
            hop_distances.append(0)

        # atoms involved only in added bonds
        elif atom in atoms_in_added_bonds:
            hop_distances.append(max_hop + 1)

        # atoms not in reaction center
        else:
            # shortest distance of atom to reaction center
            distances = []
            for center in atoms_in_reaction_center:
                try:
                    d = center_to_others_distance[center][atom]
                    distances.append(d)

                # If there are more than one reaction centers in disjoint graphs,
                # there could be no path from an atom to the center. In this case,
                # center_to_others_distance[center] does not exists for `atom`.
                except KeyError:
                    pass

            assert distances != [], (
                f"Cannot find path to reaction center for atom {atom}, this should not "
                "happen. The reaction probably has atoms not connected to others in "
                "both the reactants and the products. Please remove these atoms."
                f"Bad reaction is: {reaction.id}"
            )

            dist = min(distances)
            if dist > max_hop:
                dist = max_hop
            hop_distances.append(dist)

    return hop_distances


def get_bond_distance_to_reaction_center(
    reaction: Reaction, atom_hop_distances: Optional[List[int]] = None, max_hop: int = 3
) -> List[int]:
    """
    Get the bond hop distance (graph distance) from the reaction center.

    Reaction center is defined as the broken and added bonds.
    This is done by combining the reactants and products into a single graph,
    retaining all bonds (unchanged, lost, and added).

    A broken bond has hop distance 0;
    A bond right next to the reaction center has a hop distance of `1`;
    A bond connected to the reaction center via 1 other bond has a hop distance of
    `2`;
    ...
    A bond connected to the reaction center via max_hop-1 other bonds has a hop
    distance of `max_hop`;
    Added bonds has a hop distance of `max_hop+1`;

    Note that, an added bond should also have a hop distance of 0.
    But to distinguish from broken bonds, we give it a distance  of `max_hop+1`.
    We will use the hop distance as a label for training the model,
    so it does not matter whether it is the real distance or not as long as we can
    distinguish them.

    Args:
        reaction: reaction object
        atom_hop_distances: atom hop distances obtained by
            `get_atom_distance_to_reaction_center()`. Note, this is this provided,
            the max_hop distance used in `get_atom_distance_to_reaction_center()`
            should be the same the the one used in this function.

        max_hop: maximum number of hops allowed for unchanged bonds. Bonds farther
        away than this will all have the same distance number.

    Returns:
        Hop distances of the bonds. A list of size N (number of bonds in the reaction)
        and element i is the hop distance for bond i.

    """
    if atom_hop_distances is None:
        atom_hop_distances = get_atom_distance_to_reaction_center(reaction, max_hop)
    else:
        atom_hop_distances = copy.copy(atom_hop_distances)

    unchanged_bonds = reaction.unchanged_bonds
    lost_bonds = reaction.lost_bonds
    added_bonds = reaction.added_bonds

    # For atoms in reaction center, explicitly set the hop distance to 0.
    # This is needed since `atom_hop_distances` obtained from
    # get_atom_distance_to_reaction_center() set atoms in added bond to max_hop+1.
    atoms_in_reaction_center = set(
        [i for i in itertools.chain.from_iterable(lost_bonds + added_bonds)]
    )
    atom_hop_distances = [
        0 if atom in atoms_in_reaction_center else dist
        for atom, dist in enumerate(atom_hop_distances)
    ]

    reactants_bonds = reaction.get_reactants_bonds(zero_based=True)
    products_bonds = reaction.get_products_bonds(zero_based=True)
    reactants_bond_map_number = reaction.get_reactants_bond_map_number(for_changed=True)
    products_bond_map_number = reaction.get_products_bond_map_number(for_changed=True)

    # correspondence between bond index (atom1, atom2) and bond map number
    reactants_bond_index_to_map_number = {}
    products_bond_index_to_map_number = {}
    for bonds, map_number in zip(reactants_bonds, reactants_bond_map_number):
        for b, mn in zip(bonds, map_number):
            reactants_bond_index_to_map_number[b] = mn
    for bonds, map_number in zip(products_bonds, products_bond_map_number):
        for b, mn in zip(bonds, map_number):
            products_bond_index_to_map_number[b] = mn

    num_lost_bonds = len(lost_bonds)
    num_bonds = len(unchanged_bonds + lost_bonds + added_bonds)

    # In `combine_graphs()`, the bond node in the graph are reordered according to bond
    # map number. In `create_reaction_graph()`, the unchanged bonds will have bond
    # node number 0, 1, ... N_unchanged-1, the lost bonds in the reactants will have
    # bond node number N_unchanged, ... N-1, where N is the number of bonds in the
    # reactants, and the added bonds will have bond node number N, ... N+N_added-1.
    # We shifted the indices of the added bonds right by `the number of lost bonds`
    # to make a graph containing all bonds. Here we do the same shift for added bonds.

    hop_distances = [None] * num_bonds

    for bond in lost_bonds:
        idx = reactants_bond_index_to_map_number[bond]
        hop_distances[idx] = 0

    for bond in added_bonds:
        idx = products_bond_index_to_map_number[bond] + num_lost_bonds
        hop_distances[idx] = max_hop + 1

    for bond in unchanged_bonds:
        atom1, atom2 = bond
        atom1_hop_dist = atom_hop_distances[atom1]
        atom2_hop_dist = atom_hop_distances[atom2]

        if atom1_hop_dist == atom2_hop_dist:
            dist = atom1_hop_dist + 1
        else:
            dist = max(atom1_hop_dist, atom2_hop_dist)

        if dist > max_hop:
            dist = max_hop

        idx = reactants_bond_index_to_map_number[bond]
        hop_distances[idx] = dist

    assert None not in hop_distances, (
        "Some bond has not hop distance, this should not happen. Bad reaction is: :"
        f"{reaction.id}"
    )

    return hop_distances
