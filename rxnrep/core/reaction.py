import itertools
import logging
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from rxnrep.core.molecule import Molecule

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
    """

    def __init__(
        self,
        reactants: List[Molecule],
        products: List[Molecule],
        reagents: Optional[List[Molecule]] = None,
        sanity_check: bool = True,
        id: Optional[Union[int, str]] = None,
    ):

        self._reactants = reactants
        self._products = products
        self._reagents = reagents
        self._id = id

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
    def unchanged_bonds(self) -> List[Tuple[int, int]]:
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
    def lost_bonds(self) -> List[Tuple[int, int]]:
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
    def added_bonds(self) -> List[Tuple[int, int]]:
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
        """
        if self._species is None:
            map_number = list(
                chain.from_iterable(self.get_reactants_atom_map_number(zero_based=True))
            )
            specs = list(chain.from_iterable([m.species for m in self.reactants]))
            specs_ordered = [specs[map_number.index(i)] for i in range(len(map_number))]
            self._species = specs_ordered
        return self._species

    def get_reactants_atom_map_number(self, zero_based=False) -> List[List[int]]:
        """
        Get the atom map number of the reactant molecules.

        Args:
            zero_based: whether to convert the atom map number to zero based.
                If `True`, all atom map numbers will subtract their minimum value.

        Returns:
            Each inner list is the atom map number one reactant molecule.
        """
        return self._get_atom_map_number(self.reactants, zero_based)

    def get_products_atom_map_number(self, zero_based=False) -> List[List[int]]:
        """
        Get the atom map number of the product molecules.

        Args:
            zero_based: whether to convert the atom map number to zero based.
                If `True`, all atom map numbers will subtract their minimum value.

        Returns:
            Each inner list is the atom map number one reactant molecule.
        """
        return self._get_atom_map_number(self.products, zero_based)

    def get_reactants_bonds(
        self, zero_based: bool = False
    ) -> List[List[Tuple[int, int]]]:
        """
        Get the bonds of all the reactants.

        Args:
            zero_based: whether to convert the bond index (a pair of atom map number)
                to start from 0.

        Returns:
            Each inner list is the bonds for one reactant molecule. The bonds has the
            same order as that in the underlying rdkit molecule. Each bond is indexed as
            (atom1, atom2), and atom1 and atom2 are the two atoms forming the bonds.

        """
        return self._get_bonds(self.reactants, zero_based)

    def get_products_bonds(
        self, zero_based: bool = False
    ) -> List[List[Tuple[int, int]]]:
        """
        Get the bonds of all the products.

        Args:
            zero_based: whether to convert the bond index (a pair of atom map number)
                to start from 0.

        Returns:
            Each inner list is the bonds for one product molecule. The bonds has the
            same order as that in the underlying rdkit molecule. Each bond is indexed as
            (atom1, atom2), and atom1 and atom2 are the two atoms forming the bonds.

        """
        return self._get_bonds(self.products, zero_based)

    def get_reactants_bond_map_number(
        self, for_changed: bool = False
    ) -> List[List[int]]:
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

        Returns:
            Each inner list is the bond map number for a molecule. The bonds has the
            same order as that in the underlying rdkit molecule (i.e. element 0 is bond
            0 in the rdkit molecule).
        """
        return self._get_bond_map_number(for_changed, mode="reactants")

    def get_products_bond_map_number(
        self, for_changed: bool = False
    ) -> List[List[int]]:
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

        Returns:
            Each inner list is the bond map number for a molecule. The bonds has the
            same order as that in the underlying rdkit molecule (i.e. element 0 is bond
            0 in the rdkit molecule).
        """
        return self._get_bond_map_number(for_changed, mode="products")

    def get_unchanged_lost_and_added_bonds(
        self, zero_based: bool = False
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
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

    @staticmethod
    def _get_atom_map_number(molecules: List[Molecule], zero_based=False):
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
        molecules: List[Molecule], zero_based: bool = False
    ) -> List[List[Tuple[int, int]]]:
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
        self, for_changed=False, mode="reactants"
    ) -> List[List[int]]:
        """
        Args:
            for_changed: whether to generate bond map for changed bonds (lost bonds
                for reactants and added bonds for reactants). If `False`, the bond map
                for changed bonds are set to `None`. If `True`, their values are set to
                N_un, ..., N-1, where N_un is the number of unchanged bonds and N is the
                number of bonds.
            mode: [`reactants`|`products`]. Generate bond map for the reactant or
                the product molecules.
        """
        reactants_bonds = self.get_reactants_bonds()
        products_bonds = self.get_products_bonds()

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
            number = []  # map number for a one molecule
            for b in bonds:
                if b in unchanged_bonds:
                    number.append(bonds_map[b])
                else:
                    if for_changed:
                        number.append(bonds_map[b])
                    else:
                        number.append(None)
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

    def draw(self, filename: Path = None):
        """
        draw the reaction.

        Args:
             filename: Save to `filename` if it is not None. Example: reaction.png
        """
        rxn = AllChem.ReactionFromSmarts(str(self), useSmiles=True)
        if filename is not None:
            image = Chem.Draw.ReactionToImage(rxn)
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
    id: Optional[Union[int, str]] = None,
    ignore_reagents: bool = False,
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
        sanity_check: whether to check the correctness of the reaction
    """

    reactants, reagents, products = smiles.split(">")

    rcts = [Molecule.from_smiles(s) for s in reactants.split(".")]
    prdts = [Molecule.from_smiles(s) for s in products.split(".")]

    if ignore_reagents or reagents == "":
        rgts = None
    else:
        rgts = [Molecule.from_smiles(s) for s in reagents.split(".")]

    reaction = Reaction(rcts, prdts, rgts, sanity_check=sanity_check, id=id)

    return reaction


class ReactionError(Exception):
    def __init__(self, msg=None):
        super(ReactionError, self).__init__(msg)
        self.msg = msg
