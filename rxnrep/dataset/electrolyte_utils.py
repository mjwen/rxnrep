import warnings
import numpy as np
from collections import defaultdict
from pymatgen import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.io.babel import BabelMolAdaptor
from rdkit import Chem
from rdkit.Chem import BondType
from rdkit.Geometry import Point3D
from openbabel import openbabel as ob

from pymatgen.entries.mol_entry import MoleculeEntry

from typing import List, Tuple, Dict


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


def check_connectivity(
    mol: MoleculeEntry,
    allowed_connectivity: Dict[str, List[int]] = None,
    exclude_species: List[str] = None,
):
    """
    Check the connectivity of each atom in a mol, without considering their bonding to
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

    neigh_species = get_neighbor_species(mol)

    do_fail = False
    reason = []

    for a_s, n_s in neigh_species:

        if exclude_species is not None:
            num_bonds = len([s for s in n_s if s not in exclude_species])
        else:
            num_bonds = len(n_s)

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


def create_rdkit_mol(
    species, coords, bond_types, formal_charge=None, name=None, force_sanitize=True
):
    """
    Create a rdkit mol from scratch.

    Followed: https://sourceforge.net/p/rdkit/mailman/message/36474923/

    Args:
        species (list): species str of each molecule
        coords (2D array): positions of atoms
        bond_types (dict): with bond indices (2 tuple) as key and bond type
            (e.g. Chem.rdchem.BondType.DOUBLE) as value
        formal_charge (list): formal charge of each atom
        name (str): name of the molecule
        force_sanitize (bool): whether to force the sanitization of molecule.
            If `True` and the sanitization fails, it generally throw an error
            and then stops. If `False`, will try to sanitize first, but if it fails,
            will proceed smoothly giving a warning message.

    Returns:
        rdkit Chem.Mol
    """

    m = Chem.Mol()
    edm = Chem.EditableMol(m)
    conformer = Chem.Conformer(len(species))

    for i, (s, c) in enumerate(zip(species, coords)):
        atom = Chem.Atom(s)
        atom.SetNoImplicit(True)
        if formal_charge is not None:
            cg = formal_charge[i]
            if cg is not None:
                atom.SetFormalCharge(cg)
        atom_idx = edm.AddAtom(atom)
        conformer.SetAtomPosition(atom_idx, Point3D(*c))

    for b, t in bond_types.items():
        edm.AddBond(b[0], b[1], t)

    m = edm.GetMol()
    if force_sanitize:
        Chem.SanitizeMol(m)
    else:
        try:
            Chem.SanitizeMol(m)
        except Exception as e:
            warnings.warn(f"Cannot sanitize molecule {name}, because {str(e)}")
    m.AddConformer(conformer, assignId=False)

    m.SetProp("_Name", str(name))

    return m


def create_rdkit_mol_from_mol_graph(
    mol_graph: MoleculeGraph, name=None, force_sanitize=False, metals={"Li": 1, "Mg": 2}
):
    """
    Create a rdkit molecule from molecule graph, with bond type perceived by babel.
    Done in the below steps:

    1. create a babel mol without metal atoms.
    2. perceive bond order (conducted by BabelMolAdaptor)
    3. adjust formal charge of metal atoms so as not to violate valence rule
    4. create rdkit mol based on species, coords, bonds, and formal charge

    Args:
        mol_graph (pymatgen MoleculeGraph): molecule graph
        name (str): name of the molecule
        force_sanitize (bool): whether to force sanitization of the rdkit mol
        metals dict: with metal atom (str) as key and the number of valence electrons
            as key.

    Returns:
        m: rdkit Chem.Mol
        bond_types (dict): bond types assigned to the created rdkit mol
    """

    pymatgen_mol = mol_graph.molecule
    species = [str(s) for s in pymatgen_mol.species]
    coords = pymatgen_mol.cart_coords
    bonds = [tuple(sorted([i, j])) for i, j, attr in mol_graph.graph.edges.data()]

    # create babel mol without metals
    pmg_mol_no_metals = remove_metals(pymatgen_mol)
    adaptor = BabelMolAdaptor(pmg_mol_no_metals)
    ob_mol = adaptor.openbabel_mol

    # get babel bond order of mol without metals
    ob_bond_order = {}
    for bd in ob.OBMolBondIter(ob_mol):
        k = tuple(sorted([bd.GetBeginAtomIdx(), bd.GetEndAtomIdx()]))
        v = bd.GetBondOrder()
        ob_bond_order[k] = v

    # create bond type
    atom_idx_mapping = pymatgen_2_babel_atom_idx_map(pymatgen_mol, ob_mol)
    bond_types = {}

    for bd in bonds:
        try:
            ob_bond = [atom_idx_mapping[a] for a in bd]

            # atom not in ob mol
            if None in ob_bond:
                raise KeyError
            # atom in ob mol
            else:
                ob_bond = tuple(sorted(ob_bond))
                v = ob_bond_order[ob_bond]
                if v == 0:
                    tp = BondType.UNSPECIFIED
                elif v == 1:
                    tp = BondType.SINGLE
                elif v == 2:
                    tp = BondType.DOUBLE
                elif v == 3:
                    tp = BondType.TRIPLE
                elif v == 5:
                    tp = BondType.AROMATIC
                else:
                    raise RuntimeError(f"Got unexpected babel bond order: {v}")

        except KeyError:
            atom1_spec, atom2_spec = [species[a] for a in bd]

            if atom1_spec in metals and atom2_spec in metals:
                raise RuntimeError("Got a bond between two metal atoms")

            # bond involves one and only one metal atom (atom not in ob mol case above)
            elif atom1_spec in metals or atom2_spec in metals:
                tp = Chem.rdchem.BondType.DATIVE

                # Dative bonds have the special characteristic that they do not affect
                # the valence on the start atom, but do affect the end atom.
                # Here we adjust the atom ordering in the bond for dative bond to make
                # metal the end atom.
                if atom1_spec in metals:
                    bd = tuple(reversed(bd))

            # bond not found by babel (atom in ob mol)
            else:
                tp = Chem.rdchem.BondType.UNSPECIFIED

        bond_types[bd] = tp

    # a metal atom can form multiple dative bond (e.g. bidentate LiEC), for such cases
    # we need to adjust the their formal charge so as not to violate valence rule
    formal_charge = adjust_formal_charge(species, bonds, metals)

    m = create_rdkit_mol(
        species, coords, bond_types, formal_charge, name, force_sanitize
    )

    return m, bond_types


def adjust_formal_charge(
    species: List[str], bonds: List[Tuple[int, int]], metals: Dict[str, int]
):
    """
    Adjust formal charge of metal atoms.

    Args:
        species: species string of atoms
        bonds: 2-tuple index of bonds
        metals: initial formal charge of models

    Returns:
        list: formal charge of atoms. None for non metal atoms.
    """
    # initialize formal charge first so that atom does not form any bond has its formal
    # charge set
    formal_charge = [metals[s] if s in metals else None for s in species]

    # atom_idx: idx of atoms in the molecule
    # num_bonds: number of bonds the atom forms
    atom_idx, num_bonds = np.unique(bonds, return_counts=True)
    for i, ct in zip(atom_idx, num_bonds):
        s = species[i]
        if s in metals:
            formal_charge[i] = int(formal_charge[i] - ct)

    return formal_charge


def remove_metals(mol: Molecule, metals=None):
    """
    Check whether metals are in a pymatgen molecule. If yes, create a new Molecule
    with metals removed.

    Args:
        mol: pymatgen molecule
        metals: dict with metal specie are key and charge as value

    Returns:
        pymatgen mol
    """

    if metals is None:
        metals = {"Li": 1, "Mg": 2}

    species = [str(s) for s in mol.species]

    if set(species).intersection(set(metals.keys())):
        charge = mol.charge

        species = []
        coords = []
        properties = defaultdict(list)
        for site in mol:
            s = str(site.specie)
            if s in metals:
                charge -= metals[s]
            else:
                species.append(s)
                coords.append(site.coords)
                for k, v in site.properties:
                    properties[k].append(v)

        # do not provide spin_multiplicity, since we remove an atom
        mol = Molecule(species, coords, charge, site_properties=properties)

    return mol


def pymatgen_2_babel_atom_idx_map(pmg_mol: Molecule, ob_mol):
    """
    Create an atom index mapping between pymatgen mol and openbabel mol.

    This does not require pymatgen mol and ob mol has the same number of atoms.
    But ob_mol can have smaller number of atoms.

    Returns:
        dict: with atom index in pymatgen mol as key and atom index in babel mol as
            value. Value is `None` if there is not corresponding atom in babel.
    """

    pmg_coords = pmg_mol.cart_coords
    ob_coords = [[a.GetX(), a.GetY(), a.GetZ()] for a in ob.OBMolAtomIter(ob_mol)]
    ob_index = [a.GetIdx() for a in ob.OBMolAtomIter(ob_mol)]

    mapping = {i: None for i in range(len(pmg_coords))}

    for idx, oc in zip(ob_index, ob_coords):
        for i, gc in enumerate(pmg_coords):
            if np.allclose(oc, gc):
                mapping[i] = idx
                break
        else:
            raise RuntimeError("Cannot create atom index mapping pymatgen and ob mols")

    return mapping
