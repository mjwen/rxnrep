import copy
from collections import OrderedDict
from typing import Dict, List, Tuple, Union

import networkx as nx
import networkx.algorithms.isomorphism as iso
import pymatgen
from pymatgen.analysis.graphs import MoleculeGraph, MolGraphSplitError

from rxnrep.core.molecule import Molecule


def generate_atom_map_number_one_bond_break_reaction(
    smiles: str, *, broken_bond: int = None, add_H: bool = False
) -> str:

    reactants, _, products = smiles.split(">")

    rcts = [Molecule.from_smiles(s, remove_H=False) for s in reactants.split(".")]
    prdts = [Molecule.from_smiles(s, remove_H=False) for s in products.split(".")]

    if add_H:
        rcts = [m.add_H(explicit_only=False) for m in rcts]
        prdts = [m.add_H(explicit_only=False) for m in prdts]
    else:
        rcts = [m.remove_H(implicit_only=False) for m in rcts]
        prdts = [m.remove_H(implicit_only=False) for m in prdts]

    assert len(rcts) == 1, f"Expect 1 reactant molecule, got {len(rcts)}"
    assert len(prdts) == 1 or len(prdts) == 2, (
        f"Expect 1 or 2 product molecules, got" f" {len(prdts)}"
    )

    rct = rcts[0]
    rct_mapping, prdts_mapping = get_atom_mapping(rct, prdts, broken_bond)

    # convert to 1 based mapping
    rct_mapping = {k: v + 1 for k, v in rct_mapping.items()}
    prdts_mapping = [{k: v + 1 for k, v in mp.items()} for mp in prdts_mapping]

    rct.set_atom_map_number(rct_mapping)
    for p, m in zip(prdts, prdts_mapping):
        p.set_atom_map_number(m)

    smiles = rct.to_smiles() + ">>" + ".".join([m.to_smiles() for m in prdts])

    return smiles


Atom_Mapping_Dict = Dict[int, int]


def get_atom_mapping(
    reactant: Molecule, products: List[Molecule], broken_bond: Tuple[int, int] = None
) -> Tuple[Atom_Mapping_Dict, List[Atom_Mapping_Dict]]:

    """
    Generate rdkit style atom mapping for reactions with one reactant and one or two
    products.

    The atom mapping number for reactant atoms are simply set to their index,
    and the atom mapping number for product atoms are determined accordingly.
    Atoms in the reactant and products with the same atom mapping number (value in the
    atom mapping dictionary {atom_index: atom_mapping_number}) corresponds to each other.

    For example, given reactant

          C 0
         / \
        /___\
       O     N---H
       1     2   3

    and the two products
          C 1
         / \
        /___\
       O     N
       2     0

    and
       H 0

    This function returns:
    reactant_atom_mapping = {0:0, 1:1, 2:2, 3:3}
    products_atom_mapping = [{0:2, 1:0, 2:1}, {0:3}]

    Args:
        reactant: reactant molecule
        products: products molecule
        broken_bond: a bond in reactant, by breaking which can form the two products

    Returns:
        reactant_atom_mapping: rdkit style atom mapping number for the reactant
        products_atom_mapping: rdkit style atom mapping number for the two products
    """

    assert len(products) == 2, f"Expect 2 product molecules, got {len(products)}."

    rct_mol_graph = mol_to_pmg_mol_graph(reactant)
    prdt_mol_graphs = [mol_to_pmg_mol_graph(m) for m in products]

    if broken_bond is None:
        if len(prdt_mol_graphs) == 1:
            broken_bonds = get_broken_bonds_A_to_B(
                rct_mol_graph, prdt_mol_graphs[0], reactant.bonds, first_only=True
            )
        elif len(prdt_mol_graphs) == 2:
            broken_bonds = get_broken_bonds_A_to_B_C(
                rct_mol_graph,
                prdt_mol_graphs[0],
                prdt_mol_graphs[1],
                reactant.bonds,
                first_only=True,
            )
        else:
            raise RuntimeError("Not supported number of products")
        broken_bond = broken_bonds[0]

    # Split the reactant mol graph to form two sub graphs
    # This is similar to MoleculeGraph.split_molecule_subbraphs(), but do not reorder
    # the nodes, i.e. the nodes in the subgraphs will have the same node indexes as
    rct_mol_graph.break_edge(broken_bond[0], broken_bond[1], allow_reverse=True)
    components = nx.weakly_connected_components(rct_mol_graph.graph)
    sub_graphs = [rct_mol_graph.graph.subgraph(c) for c in components]

    # correspondence between products and reactant sub graphs
    if len(sub_graphs) == 1:
        corr = {0: 0}

    else:
        # product idx as key and reactant subgraph idx as value
        # order matters since mappings[0] (see below) corresponds to first product
        corr = OrderedDict()

        is_iso, _ = is_isomorphic(prdt_mol_graphs[0].graph, sub_graphs[0])
        if is_iso:
            # product 0 corresponds to sub graph 0, and product 1 to sub graph 1
            corr[0] = 0
            corr[1] = 1
        else:
            # product 0 corresponds to sub graph 1, and product 1 to sub graph 0
            corr[0] = 1
            corr[1] = 0

    products_atom_mapping = []
    for i, j in corr.items():
        _, node_mapping = is_isomorphic(prdt_mol_graphs[i].graph, sub_graphs[j])
        assert (
            node_mapping is not None
        ), "Cannot obtain node mapping. Should not happen."
        products_atom_mapping.append(node_mapping)

    reactant_atom_mapping = {i: i for i in range(reactant.num_atoms)}

    return reactant_atom_mapping, products_atom_mapping


def mol_to_pmg_mol_graph(m: Molecule) -> MoleculeGraph:
    """
    Convert a rxnrep molecule to pymatgen molecule graph.
    """
    pmg_m = pymatgen.Molecule(m.species, m.coords, m.charge)
    bonds = {tuple(sorted(b)): None for b in m.bonds}

    mol_graph = MoleculeGraph.with_edges(pmg_m, bonds)

    return mol_graph


def get_broken_bonds_A_to_B(
    reactant: MoleculeGraph,
    product: MoleculeGraph,
    bonds: List[Tuple[int, int]],
    first_only: bool = True,
) -> List[Tuple[int, int]]:
    """
    Check whether the reactant and product can form A -> B style reaction by breaking
    one bond in the reactant. Return the broken bonds that lead.

    Args:
        reactant: molecule graph
        product: molecule graph
        bonds: all bonds in the molecule.
        first_only: If `True`, only return the first found such bond; otherwise return
            all.

    Returns:
        Bonds of the reactant (represented by a tuple of the two atoms associated
        with the bond) by breaking which A -> B reaction is valid. Empty if no such
        reaction can form.
    """
    broken_bonds = []
    fragments = fragment_mol_graph(reactant, bonds)
    for b, mgs in fragments.items():
        if len(mgs) == 1 and mgs[0].isomorphic_to(product):
            broken_bonds.append(b)
            if first_only:
                return broken_bonds
    return broken_bonds


def get_broken_bonds_A_to_B_C(
    reactant: MoleculeGraph,
    product1: MoleculeGraph,
    product2: MoleculeGraph,
    bonds: List[Tuple[int, int]],
    first_only: bool = True,
):
    """
    Check whether the reactant and product can form A -> B style reaction by breaking
    one bond in the reactant. Return the broken bonds that lead.

    Args:
        reactant: molecule graph
        product1: molecule graph
        product2: molecule graph
        bonds: all bonds in the molecule.
        first_only: If `True`, only return the first found such bond; otherwise return
            all.

    Returns:
        Bonds of the reactant (represented by a tuple of the two atoms associated
        with the bond) by breaking which A -> B reaction is valid. Empty if no such
        reaction can form.

    """

    broken_bonds = []
    fragments = fragment_mol_graph(reactant, bonds)

    for b, mgs in fragments.items():
        if len(mgs) == 2:
            if (mgs[0].isomorphic_to(product1) and mgs[1].isomorphic_to(product2)) or (
                mgs[0].isomorphic_to(product2) and mgs[1].isomorphic_to(product1)
            ):
                broken_bonds.append(b)
                if first_only:
                    return broken_bonds

    return broken_bonds


def fragment_mol_graph(
    mol_graph: MoleculeGraph, bonds: List[Tuple[int, int]]
) -> Dict[Tuple[int, int], List[MoleculeGraph]]:
    """
    Break a bond in molecule graph and obtain the fragment(s).

    Args:
        mol_graph: molecule graph to fragment
        bonds: bonds to break

    Returns:
        Fragments of the molecule. A dict with key the bond index specified in bonds,
        and value a list of fragments obtained by breaking the bond.
        Each list could be of size 1 or 2 and could be empty if the mol has no bonds.
    """
    fragments = {}

    for edge in bonds:
        edge = tuple(edge)
        try:
            # breaking a bond generates two fragments
            new_mgs = mol_graph.split_molecule_subgraphs(
                [edge], allow_reverse=True, alterations=None
            )
            fragments[edge] = new_mgs
        except MolGraphSplitError:
            # breaking a bond generates one fragment, i.e. breaking bone in a ring
            new_mg = copy.deepcopy(mol_graph)
            idx1, idx2 = edge
            new_mg.break_edge(idx1, idx2, allow_reverse=True)
            fragments[edge] = [new_mg]
    return fragments


def is_isomorphic(
    g1: nx.MultiDiGraph, g2: nx.MultiDiGraph
) -> Tuple[bool, Union[None, Dict[int, int]]]:
    """
    Check the isomorphic between two graphs g1 and g2 and return the node mapping.

    Args:
        g1: nx graph
        g2: nx graph

    See Also:
        https://networkx.github.io/documentation/stable/reference/algorithms/isomorphism.vf2.html

    Returns:
        is_isomorphic: Whether graphs g1 and g2 are isomorphic.
        node_mapping: Node mapping from g1 to g2 (e.g. {0:2, 1:1, 2:0}), if g1 and g2
            are isomorphic, `None` if not isomorphic.
    """
    nm = iso.categorical_node_match("specie", "ERROR")
    GM = iso.GraphMatcher(g1.to_undirected(), g2.to_undirected(), node_match=nm)
    if GM.is_isomorphic():
        return True, GM.mapping
    else:
        return False, None
