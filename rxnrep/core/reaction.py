import itertools
import logging
from collections import OrderedDict, defaultdict
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from rxnrep.core.molecule import Molecule
from rxnrep.typing import BondIndex

logger = logging.getLogger(__name__)


class Reaction:
    """
    A reaction with reactants, products and optionally reagents.

    Args:
        reactants: reactants of a reaction
        products: products of a reaction
        reagents: optional reagents of a reaction
        sanity_check: check the correctness of the reactions, e.g. mass conservation,
            charge conservation...
        id: an identifier of the reaction
        properties: a dictionary of additional properties associated with the reaction,
            e.g. reaction energy
    """

    def __init__(
        self,
        reactants: List[Molecule],
        products: List[Molecule],
        reagents: Optional[List[Molecule]] = None,
        sanity_check: bool = True,
        id: Optional[Union[int, str]] = None,
        properties: Optional[Dict[str, Any]] = None,
    ):

        self._reactants = reactants
        self._products = products
        self._reagents = reagents
        self._id = id
        self._properties = properties

        self._unchanged_bonds = None
        self._lost_bonds = None
        self._added_bonds = None

        self._species = None

        if sanity_check:
            self.check_composition()
            self.check_charge()
            self.check_atom_map_number()

    @property
    def reactants(self) -> List[Molecule]:
        return self._reactants

    @property
    def products(self) -> List[Molecule]:
        return self._products

    @property
    def reagents(self) -> Union[List[Molecule], None]:
        return self._reagents

    @property
    def id(self) -> Union[int, str, None]:
        """
        Returns the identifier of the reaction.
        """
        return self._id

    @property
    def unchanged_bonds(self) -> List[BondIndex]:
        """
        Unchanged bonds, i.e. bonds exist in both the reactants and products.
        Each bond is indexed by atom map number of the two atoms forming the bond.
        """
        if self._unchanged_bonds is None:
            (
                self._unchanged_bonds,
                self._lost_bonds,
                self._added_bonds,
            ) = self.get_unchanged_lost_and_added_bonds(zero_based=True)

        return self._unchanged_bonds

    @property
    def lost_bonds(self) -> List[BondIndex]:
        """
        Lost bonds, i.e. bonds in reactants but not in products.
        Each bond is indexed by atom map number of the two atoms forming the bond.
        """
        if self._lost_bonds is None:
            (
                self._unchanged_bonds,
                self._lost_bonds,
                self._added_bonds,
            ) = self.get_unchanged_lost_and_added_bonds(zero_based=True)

        return self._lost_bonds

    @property
    def added_bonds(self) -> List[BondIndex]:
        """
        Added bonds, i.e. bonds in products but not in reactants.
        Each bond is indexed by atom map number of the two atoms forming the bond.
        """
        if self._added_bonds is None:
            (
                self._unchanged_bonds,
                self._lost_bonds,
                self._added_bonds,
            ) = self.get_unchanged_lost_and_added_bonds(zero_based=True)

        return self._added_bonds

    @property
    def species(self) -> List[str]:
        """
        Get the species of the atoms, ordered according to the atom map number.

        This assumes the atom map number are contiguous, e.g. from 0 to the number of
        atoms minus 1.
        """
        if self._species is None:
            map_number = list(
                chain.from_iterable(self.get_reactants_atom_map_number(zero_based=True))
            )
            specs = list(chain.from_iterable([m.species for m in self.reactants]))
            specs_ordered = [specs[map_number.index(i)] for i in range(len(map_number))]
            self._species = specs_ordered
        return self._species

    def get_reactants_atom_map_number(self, zero_based=True) -> List[List[int]]:
        """
        Get the atom map number of the reactant molecules.

        Args:
            zero_based: whether to convert the atom map number to zero based.
                If `True`, all atom map numbers will subtract their minimum value.

        Returns:
            Each inner list is the atom map number one reactant molecule.
        """
        return self._get_atom_map_number(self.reactants, zero_based)

    def get_products_atom_map_number(self, zero_based=True) -> List[List[int]]:
        """
        Get the atom map number of the product molecules.

        Args:
            zero_based: whether to convert the atom map number to zero based.
                If `True`, all atom map numbers will subtract their minimum value.

        Returns:
            Each inner list is the atom map number one reactant molecule.
        """
        return self._get_atom_map_number(self.products, zero_based)

    def get_reactants_bonds(self, zero_based: bool = True) -> List[List[BondIndex]]:
        """
        Get the bonds of all the reactants.

        Each bond is indexed by the atom map number of the two atoms forming the bond.

        Args:
            zero_based: whether to convert the bond index (atom map number) to 0 based.

        Returns:
            Each inner list is the bonds for one reactant molecule. The bonds has the
            same order as that in the underlying rdkit molecule. Each bond is indexed as
            (amn1, amn2), where amn1 and amn2 are the atom map number of the two atoms
            forming the bond.
        """
        return self._get_bonds(self.reactants, zero_based)

    def get_products_bonds(self, zero_based: bool = True) -> List[List[BondIndex]]:
        """
        Get the bonds of all the products.

        Each bond is indexed by the atom map number of the two atoms forming the bond.

        Args:
            zero_based: whether to convert the bond index (atom map number) to 0 based.

        Returns:
            Each inner list is the bonds for one product molecule. The bonds has the
            same order as that in the underlying rdkit molecule. Each bond is indexed as
            (amn1, amn2), where amn1 and amn2 are the atom map number of the two atoms
            forming the bond.
        """
        return self._get_bonds(self.products, zero_based)

    def get_reactants_bond_map_number(
        self, for_changed: bool = False, as_dict: bool = False
    ) -> Union[List[List[int]], Dict[BondIndex, int]]:
        """
        Get the bond map number of the reactant molecules.

        The bonds are divided into two categories:
        1) unchanged bonds: bonds exist in both the reactants and the products.
        2) lost bonds: bonds in the reactants but not in the products.

        For unchanged bonds, the bond map numbers are 0,1, ..., N_un-1, where N_un is
        the number of unchanged bonds. The bond map number for changed bonds depends on
        the value of `for_changed`. See below.

        Args:
            for_changed: whether to generate bond map for changed bonds (lost bonds
                for reactants and added bonds for reactants). If `False`, the bond map
                for changed bonds are set to `None`. If `True`, their values are set to
                N_un, ..., N-1, where N_un is the number of unchanged bonds and N is the
                number of bonds. Note, although we assigned a bond map number to it,
                it does not mean the bond corresponds to the bond in the products with
                the same map number.
            as_dict: determines the format of the return. see below.

        Returns:
            If `as_dict = False`, each inner list is the bond map number for a molecule.
            the bonds has the same order as that in the underlying rdkit molecule
            (i.e. element 0 is bond 0 in the rdkit molecule).
            If `as_dict = True`, each inner dict is the bond map number of a molecule:
            {bond_index, map_number}, where bond_index is a tuple of the atom map
            number of the two atoms forming the bond.
        """
        return self._get_bond_map_number("reactants", for_changed, as_dict)

    def get_products_bond_map_number(
        self, for_changed: bool = False, as_dict: bool = False
    ) -> Union[List[List[int]], Dict[BondIndex, int]]:
        """
        Get the bond map number of the product molecules.

        The bonds are divided into two categories:
        1) unchanged bonds: bonds exist in both the reactants and the products.
        2) added bonds: bonds not in the reactants but in the products.

        For unchanged bonds, the bond map numbers are 0,1, ..., N_un -1, where N_un is
        the number of unchanged bonds. The bond map number for changed bonds depends on
        the value of `for_changed`. See below.

        Args:
            for_changed: whether to generate bond map for changed bonds (lost bonds
                for reactants and added bonds for reactants). If `False`, the bond map
                for changed bonds are set to `None`. If `True`, their values are set to
                N_un, ..., N-1, where N_un is the number of unchanged bonds and N is the
                number of bonds. Note, although we assigned a bond map number to it,
                it does not mean the bond corresponds to the bond in the reactants with
                the same map number.
            as_dict: determines the format of the return. see below.

        Returns:
            If `as_dict = False`, each inner list is the bond map number for a molecule.
            the bonds has the same order as that in the underlying rdkit molecule
            (i.e. element 0 is bond 0 in the rdkit molecule).
            If `as_dict = True`, each inner dict is the bond map number of a molecule:
            {bond_index, map_number}, where bond_index is a tuple of the atom map
            number of the two atoms forming the bond.
        """
        return self._get_bond_map_number("products", for_changed, as_dict)

    def get_unchanged_lost_and_added_bonds(
        self, zero_based: bool = True
    ) -> Tuple[List[BondIndex], List[BondIndex], List[BondIndex]]:
        """
        Get the unchanged, lost, and added bonds in the reaction.

        `unchanged` means the connectivity of the molecules. So, this includes the
        bonds exist in both the reactants and the products. If a bond changes its
        bond type (e.g. from single to double), it is still considered as unchanged.
        `lost` means the bonds in the reactants but not in the products.
        `added` means the bonds in the products but not in the reactants.

        Each bond is index as (atom1, atom2) where atom1 and atom2 are the atom map
        numbers of the two atoms forming the two bonds.
        Args:
            zero_based: whether to convert the bond index to zero based.
                If `True`, all bond indices (atom map number) are converted to zero based.

        Returns:
            (unchanged_bonds, lost_bonds, added_bonds). Each is a list of bond indices,
                each bond index is given as a tuple (atom1, atom2).
        """
        reactants_bonds = self.get_reactants_bonds(zero_based)
        products_bonds = self.get_products_bonds(zero_based)
        reactant_bonds_set = set(itertools.chain.from_iterable(reactants_bonds))
        product_bonds_set = set(itertools.chain.from_iterable(products_bonds))

        unchanged_bonds = reactant_bonds_set & product_bonds_set
        lost_bonds = reactant_bonds_set - unchanged_bonds
        added_bonds = product_bonds_set - unchanged_bonds

        unchanged_bonds = list(unchanged_bonds)
        lost_bonds = list(lost_bonds)
        added_bonds = list(added_bonds)

        return unchanged_bonds, lost_bonds, added_bonds

    def get_altered_bond_type(self) -> str:
        """
        Return the altered bond type as a string.

        Returns:
            altered bonds: e.g. `+C,C;-C,N` means a carbon carbon bond is created and a
                carbon nitrogen bond is broken.
        """

        species = self.species
        bonds = []
        for b in self.lost_bonds:
            bonds.append("-" + ",".join(sorted([species[b[0]], species[b[1]]])))
        for b in self.added_bonds:
            bonds.append("+" + ",".join(sorted([species[b[0]], species[b[1]]])))

        altered_bonds = ";".join(sorted(bonds))

        return altered_bonds

    def set_property(self, name: str, value: Any):
        """
        Add additional property to the reaction.

        If the property is already there this will reset it.

        Args:
            name: name of the property
            value: value of the property
        """
        if self._properties is None:
            self._properties = {}

        self._properties[name] = value

    def get_property(self, name: str) -> Any:
        """
        Return the additional properties of the reaction.

        Args:
            name: property name
        """
        if self._properties is None:
            raise ReactionError(f"Reaction does not have property {name}")
        else:
            try:
                return self._properties[name]
            except KeyError:
                raise ReactionError(f"Reaction does not have property {name}")

    @staticmethod
    def _get_atom_map_number(molecules: List[Molecule], zero_based=True):
        """
        `molecules` typically should be all the reactants or products.
        """
        atom_map_number = [m.get_atom_map_number() for m in molecules]
        if zero_based:
            minimum = int(np.min(np.concatenate(atom_map_number)))
            atom_map_number = [[x - minimum for x in num] for num in atom_map_number]

        return atom_map_number

    @staticmethod
    def _get_bonds(
        molecules: List[Molecule], zero_based: bool = True
    ) -> List[List[BondIndex]]:
        """
        Get all the bonds in the molecules. Each bond is index as (atom1, atom2),
        where atom1 and atom2 are the atom map number for that atom.
        Each inner list is the bonds for one molecule, and the bonds have the same
        order as that in the underlying rdkit molecule.

        Args:
            zero_based: whether to convert the bond index to zero based.
                If `True`, all bond indices (atom map number) are converted to zero based.
        """
        all_bonds = []
        for m in molecules:
            atom_map = m.get_atom_map_number()
            all_bonds.append(
                [tuple(sorted([atom_map[b[0]], atom_map[b[1]]])) for b in m.bonds]
            )

        has_bond = False
        for bonds in all_bonds:
            if bonds:
                has_bond = True
                break

        if zero_based and has_bond:
            # smallest atom index
            atom_map_numbers = [m.get_atom_map_number() for m in molecules]
            smallest = min(itertools.chain.from_iterable(atom_map_numbers))
            # adjust bond indices
            all_bonds = [
                [(b[0] - smallest, b[1] - smallest) for b in bonds]
                for bonds in all_bonds
            ]

        return all_bonds

    def _get_bond_map_number(
        self, mode="reactants", for_changed=False, as_dict: bool = False
    ) -> List[Union[List[int], Dict[BondIndex, int]]]:
        """
        Args:
            mode: [`reactants`|`products`]. Generate bond map for the reactant or
                the product molecules.
            for_changed: whether to generate bond map for changed bonds (lost bonds
                for reactants and added bonds for reactants). If `False`, the bond map
                for changed bonds are set to `None`. If `True`, their values are set to
                N_un, ..., N-1, where N_un is the number of unchanged bonds and N is the
                number of bonds.
            as_dict: how is the bond indexed. If `False`, the map number of each
                molecule is a list of int, with the same order of bond in the molecule.
                If `True`, each bond is indexed by the two atoms from the bond,
                and the returned value for each molecule is a dict with the index as
                key and the map number as value.
        """
        reactants_bonds = self.get_reactants_bonds(zero_based=True)
        products_bonds = self.get_products_bonds(zero_based=True)

        reactant_bonds_set = set(itertools.chain.from_iterable(reactants_bonds))
        product_bonds_set = set(itertools.chain.from_iterable(products_bonds))
        unchanged_bonds = reactant_bonds_set & product_bonds_set

        if mode == "reactants":
            changed_bonds = reactant_bonds_set - unchanged_bonds
            target_bonds = reactants_bonds
        elif mode == "products":
            changed_bonds = product_bonds_set - unchanged_bonds
            target_bonds = products_bonds
        else:
            raise ValueError("not supported mode")

        # the unique bond map number is generated as the index of the sorted array
        sorted_bonds = sorted(unchanged_bonds) + sorted(changed_bonds)
        bonds_map = {v: k for k, v in enumerate(sorted_bonds)}

        bond_map_number = []  # map number for all molecules
        for bonds in target_bonds:

            number = OrderedDict()  # map number for a one molecule
            for b in bonds:
                if b in unchanged_bonds:
                    number[b] = bonds_map[b]
                else:
                    if for_changed:
                        number[b] = bonds_map[b]
                    else:
                        number[b] = None

            if not as_dict:
                number = list(number.values())
            bond_map_number.append(number)

        return bond_map_number

    def check_composition(self):
        """
        Check that composition is balanced between the reactants and products.
        """
        reactants_comp = defaultdict(int)
        products_comp = defaultdict(int)
        for m in self._reactants:
            for s in m.species:
                reactants_comp[s] += 1

        for m in self._products:
            for s in m.species:
                products_comp[s] += 1

        if reactants_comp != products_comp:
            rc = ""
            for s in sorted(reactants_comp.keys()):
                rc += f"{s}{reactants_comp[s]}"
            pc = ""
            for s in sorted(products_comp.keys()):
                pc += f"{s}{products_comp[s]}"

            raise ReactionError(
                f"Failed `check_composition()` for reaction `{self.id}`. "
                f"Reactants composition is {rc}, while products composition is {pc}."
            )

    def check_charge(self):
        """
        Check that charge is balanced between the reactants and products.
        """
        reactants_charge = sum([m.charge for m in self._reactants])
        products_charge = sum([m.charge for m in self._products])
        if reactants_charge != products_charge:
            raise ReactionError(
                f"Failed `check_charge()` for reaction `{self.id}`. "
                f"The sum of reactant charges ({reactants_charge}) "
                f"does not equal the sum of product charges ({products_charge})."
            )

    def check_atom_map_number(self):
        """
        Check the correctness of atom map number: every reactant and product atom is
        mapped, and each should have one and only one map.
        """
        reactants_map = []
        products_map = []

        # check every reactant is mapped
        for i, m in enumerate(self.reactants):
            map_number = m.get_atom_map_number()
            if None in map_number:
                raise ReactionError(
                    f"Failed `check_atom_map_number()` for reaction `{self.id}`. "
                    f"Reactant {i} has atoms without atom map number."
                )
            reactants_map.extend(map_number)

        # check every product is mapped
        for i, m in enumerate(self.products):
            map_number = m.get_atom_map_number()
            if None in map_number:
                raise ReactionError(
                    f"Failed `check_atom_map_number()` for reaction `{self.id}`. "
                    f"Products {i} has atoms without atom map number."
                )
            products_map.extend(map_number)

        # check the map is unique
        if len(reactants_map) != len(set(reactants_map)):
            raise ReactionError(
                f"Failed `check_atom_map_number()` for reaction `{self.id}`. "
                f"Reactants have atoms with the same map number."
            )
        if len(products_map) != len(set(products_map)):
            raise ReactionError(
                f"Failed `check_atom_map_number()` for reaction `{self.id}`. "
                f"Products have atoms with the same map number."
            )
        if set(reactants_map) != set(products_map):
            raise ReactionError(
                f"Failed `check_atom_map_number()` for reaction `{self.id}`. "
                f"Reactants and products have different map numbers."
            )

    def draw(self, filename: Path = None, **kwargs):
        """
        draw the reaction.

        Args:
             filename: Save to `filename` if it is not None. Example: reaction.png
             kwargs: additional kw arguments for `ReactionToImage`.
        """
        rxn = AllChem.ReactionFromSmarts(str(self), useSmiles=True)
        if filename is not None:
            image = Chem.Draw.ReactionToImage(rxn, **kwargs)
            image.save(str(filename))
        return rxn

    def __str__(self):
        """Smiles representation of reaction."""
        reactants = ".".join([m.to_smiles() for m in self.reactants])
        products = ".".join([m.to_smiles() for m in self.products])
        if self.reagents is not None:
            reagents = ".".join([m.to_smiles() for m in self.reagents])
        else:
            reagents = ""
        smiles = ">".join([reactants, reagents, products])

        return smiles


def smiles_to_reaction(
    smiles: str,
    *,
    id: Optional[Union[int, str]] = None,
    ignore_reagents: bool = False,
    remove_H: bool = True,
    sanity_check: bool = True,
):
    """
    Convert a reaction given in smiles to :class:`Reaction`.

    Args:
        smiles: a smiles representation of a reaction, where the reactants, reagents,
            and products are separated by `>`. For example:
            '[C:1](=[O:2])-[OD1].[N!H0:3]>>[C:1](=[O:2])[N:3]'
        id: identifier of the reaction.
        ignore_reagents: whether to ignore reagents, regardless of its existence
        remove_H: whether to remove H atoms.
        sanity_check: whether to check the correctness of the reaction
    """

    reactants, reagents, products = smiles.split(">")

    rcts = [Molecule.from_smiles(s, remove_H=remove_H) for s in reactants.split(".")]
    prdts = [Molecule.from_smiles(s, remove_H=remove_H) for s in products.split(".")]

    if ignore_reagents or reagents == "":
        rgts = None
    else:
        rgts = [Molecule.from_smiles(s, remove_H=remove_H) for s in reagents.split(".")]

    reaction = Reaction(rcts, prdts, rgts, sanity_check=sanity_check, id=id)

    return reaction


class ReactionError(Exception):
    def __init__(self, msg=None):
        super(ReactionError, self).__init__(msg)
        self.msg = msg
