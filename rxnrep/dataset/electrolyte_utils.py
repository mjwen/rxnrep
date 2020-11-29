from typing import List, Tuple, Dict
import numpy as np
import networkx as nx

from mrnet.core.mol_entry import MoleculeEntry
from mrnet.core.reactions import bucket_mol_entries, Reaction

from rxnrep.core.rdmol import create_rdkit_mol_from_mol_graph


def check_connectivity(mol: MoleculeEntry) -> Tuple[bool, None]:
    """
    Check whether all atoms in a molecule is connected.

    Args:
        mol:
    Returns:
    """

    # all atoms are connected, not failing
    if nx.is_weakly_connected(mol.graph):
        return False, None

    # some atoms are not connected, failing
    else:
        return True, None


def check_species(mol: MoleculeEntry, species: List[str]) -> Tuple[bool, str]:
    """
    Check whether molecule contains species given in `species`.

    Args:
        mol:
        species:

    Returns:
    """
    for s in species:
        if s in mol.species:
            return True, s
    return False, None


def check_bond_species(
    mol: MoleculeEntry,
    bond_species=(("Li", "H"), ("Li", "Li"), ("Mg", "Mg"), ("H", "Mg")),
):
    """
    Check whether molecule contains bonds with species specified in `bond_species`.
    """

    def get_bond_species(m):
        """
        Returns:
            A list of the two species associated with each bonds in the molecule.
        """
        res = []
        for a1, a2 in m.bonds:
            s1 = m.species[a1]
            s2 = m.species[a2]
            res.append(sorted([s1, s2]))
        return res

    bond_species = [sorted(i) for i in bond_species]

    mol_bond_species = get_bond_species(mol)

    contains = False
    reason = []
    for b in mol_bond_species:
        if b in bond_species:
            reason.append(b)
            contains = True

    return contains, reason


def check_bond_length(mol: MoleculeEntry, bond_length_limit=None):
    """
    Check the length of bonds. If larger than allowed length, it fails.
    """

    def get_bond_lengths(m):
        """
        Returns:
            A list of tuple (species, length), where species are the two species
            associated with a bond and length is the corresponding bond length.
        """
        coords = m.molecule.cart_coords
        res = []
        for a1, a2 in m.bonds:
            s1 = m.species[a1]
            s2 = m.species[a2]
            c1 = np.asarray(coords[a1])
            c2 = np.asarray(coords[a2])
            length = np.linalg.norm(c1 - c2)
            res.append((tuple(sorted([s1, s2])), length))
        return res

    #
    # bond lengths references:
    # http://chemistry-reference.com/tables/Bond%20Lengths%20and%20Enthalpies.pdf
    # page 29 https://slideplayer.com/slide/17256509/
    # https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Supplemental_Modules_(Physical_and_Theoretical_Chemistry)/Chemical_Bonding/Fundamentals_of_Chemical_Bonding/Chemical_Bonds/Bond_Lengths_and_Energies
    #
    # unit: Angstrom
    #

    if bond_length_limit is None:
        li_len = 2.8
        mg_len = 2.8
        bond_length_limit = {
            # H
            # ("H", "H"): 0.74,
            ("H", "H"): None,
            ("H", "C"): 1.09,
            ("H", "O"): 0.96,
            # ("H", "F"): 0.92,
            ("H", "F"): None,
            ("H", "P"): 1.44,
            ("H", "N"): 1.01,
            ("H", "S"): 1.34,
            # ("H", "Cl"): 1.27,
            ("H", "Cl"): None,
            ("H", "Li"): li_len,
            ("H", "Mg"): mg_len,
            # C
            ("C", "C"): 1.54,
            ("C", "O"): 1.43,
            ("C", "F"): 1.35,
            ("C", "P"): 1.84,
            ("C", "N"): 1.47,
            ("C", "S"): 1.81,
            ("C", "Cl"): 1.77,
            ("C", "Li"): li_len,
            ("C", "Mg"): mg_len,
            # O
            ("O", "O"): 1.48,
            ("O", "F"): 1.42,
            ("O", "P"): 1.63,
            ("O", "N"): 1.44,
            ("O", "S"): 1.51,
            ("O", "Cl"): 1.64,
            ("O", "Li"): li_len,
            ("O", "Mg"): mg_len,
            # F
            # ("F", "F"): 1.42,
            ("F", "F"): None,
            ("F", "P"): 1.54,
            ("F", "N"): 1.39,
            ("F", "S"): 1.58,
            # ("F", "Cl"): 1.66,
            ("F", "Cl"): None,
            ("F", "Li"): li_len,
            ("F", "Mg"): mg_len,
            # P
            ("P", "P"): 2.21,
            ("P", "N"): 1.77,
            ("P", "S"): 2.1,
            ("P", "Cl"): 204,
            ("P", "Li"): li_len,
            ("P", "Mg"): mg_len,
            # N
            ("N", "N"): 1.46,
            ("N", "S"): 1.68,
            ("N", "Cl"): 1.91,
            ("N", "Li"): li_len,
            ("N", "Mg"): mg_len,
            # S
            ("S", "S"): 2.04,
            ("S", "Cl"): 201,
            ("S", "Li"): li_len,
            ("S", "Mg"): mg_len,
            # Cl
            # ("Cl", "Cl"): 1.99,
            ("Cl", "Cl"): None,
            ("Cl", "Li"): li_len,
            ("Cl", "Mg"): mg_len,
            # Li
            ("Li", "Li"): li_len,
            # Mg
            ("Mg", "Mg"): mg_len,
        }

        # multiply by 1.2 to relax the rule a bit
        tmp = dict()
        for k, v in bond_length_limit.items():
            if v is not None:
                v *= 1.2
            tmp[tuple(sorted(k))] = v
        bond_length_limit = tmp

    do_fail = False
    reason = []

    bond_lengths = get_bond_lengths(mol)
    for b, length in bond_lengths:
        limit = bond_length_limit[b]
        if limit is not None and length > limit:
            reason.append("{}  {} ({})".format(b, length, limit))
            do_fail = True

    return do_fail, reason


def check_num_bonds(
    mol: MoleculeEntry,
    allowed_connectivity: Dict[str, List[int]] = None,
    exclude_species: List[str] = None,
):
    """
    Check the number of bonds of each atom in a mol, without considering their bonding to
    metal element (e.g. Li), which forms coordinate bond with other atoms.

    If there are atoms violate the connectivity specified in allowed_connectivity,
    returns True.

    Args:
        mol:
        allowed_connectivity: {specie, [connectivity]}. Allowed connectivity by specie.
            If None, use internally defined connectivity.
        exclude_species: bond formed with species given in this list are ignored
            when counting the connectivity of an atom.
    """

    def get_neighbor_species(m):
        """
        Returns:
            A list of tuple (atom species, bonded atom species),
            where `bonded_atom_species` is a list.
            Each tuple represents an atom and its bonds.
        """
        res = [(s, []) for s in m.species]
        for a1, a2 in m.bonds:
            s1 = m.species[a1]
            s2 = m.species[a2]
            res[a1][1].append(s2)
            res[a2][1].append(s1)
        return res

    if allowed_connectivity is None:
        allowed_connectivity = {
            "H": [1],
            "C": [1, 2, 3, 4],
            "O": [1, 2],
            "F": [1],
            "P": [1, 2, 3, 5, 6],  # 6 for LiPF6
            "N": [1, 2, 3, 4, 5],
            "S": [1, 2, 3, 4, 5, 6],
            "Cl": [1],
            # metal
            "Li": [1, 2, 3],
            "Mg": [1, 2, 3, 4, 5],
        }

    exclude_species = ["Li", "Mg"] if exclude_species is None else exclude_species

    neigh_species = get_neighbor_species(mol)

    do_fail = False
    reason = []

    for a_s, n_s in neigh_species:
        num_bonds = len([s for s in n_s if s not in exclude_species])

        if num_bonds == 0:  # fine since we removed metal coordinate bonds
            continue

        if num_bonds not in allowed_connectivity[a_s]:
            reason.append("{} {}".format(a_s, num_bonds))
            do_fail = True

    return do_fail, reason


def check_bad_rdkit_molecule(mol: MoleculeEntry):
    """
    Check whether a molecule is a bad molecule that cannot be converted to a rdkit
    molecule.
    """
    try:
        create_rdkit_mol_from_mol_graph(mol.mol_graph, force_sanitize=True)
        return False, None
    except Exception as e:
        return True, str(e)


def remove_high_energy_mol_entries(
    mol_entries: List[MoleculeEntry],
) -> List[MoleculeEntry]:
    """
    For molecules of the same isomorphism and charge, remove the ones with higher free
    energies.

    Args:
        mol_entries: a list of molecule entries

    Returns:
        low_energy_entries: molecule entries with high free energy ones removed
    """

    # convert list of entries to nested dicts
    buckets = bucket_mol_entries(mol_entries, keys=["formula", "num_bonds", "charge"])

    all_entries = []
    for formula in buckets:
        for num_bonds in buckets[formula]:
            for charge in buckets[formula][num_bonds]:

                # filter mols having the same formula, number bonds, and charge
                low_energy_entries = []
                for entry in buckets[formula][num_bonds][charge]:

                    # try to find an entry_i with the same isomorphism to entry
                    idx = -1
                    for i, entry_i in enumerate(low_energy_entries):
                        if entry.mol_graph.isomorphic_to(entry_i.mol_graph):
                            idx = i
                            break

                    if idx >= 0:
                        # entry has the same isomorphism as entry_i
                        if (
                            entry.get_free_energy()
                            < low_energy_entries[idx].get_free_energy()
                        ):
                            low_energy_entries[idx] = entry

                    else:
                        # entry with a unique isomorphism
                        low_energy_entries.append(entry)

                all_entries.extend(low_energy_entries)

    return all_entries


def remove_redundant_reactions(reactions: List[Reaction]) -> List[Reaction]:
    """
    Remove redundant reactions that have the same reactants and products.

    The redundant reactions may happen when there is symmetry in the molecule. For
    example, breaking any C-C bond in a benzene molecule will generate the same products.

    Warnings:
        This is achieved by comparing the entry ids. Only one reaction with the same
        reaction and product ids are kept. This function should work for one-bond
        breaking reactions (reactions with a single reactant molecule). For reactions
        with more than one bond edits, this may not remove non-redundant reactions
        since there may have different combinations by bond edits. In this case,
        this function should not be used.

    Args:
        reactions: a list of reactions

    Returns:
        a list of reactions, with redundant ones removed
    """

    filtered_reactions = []
    seen_entry_ids = []
    for rxn in reactions:
        entry_ids = (set(rxn.reactant_ids), set(rxn.product_ids))
        if entry_ids not in seen_entry_ids:
            seen_entry_ids.append(entry_ids)
            filtered_reactions.append(rxn)

    return filtered_reactions
