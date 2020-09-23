import logging
import itertools
import numpy as np
from collections import defaultdict
from rxnrep.core.molecule import Molecule
from typing import List, Tuple, Dict, Optional, Union

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

    def get_reactants_atom_map_number(self, zero_based=False) -> List[List[int]]:
        """
        Get the atom map number of the reactant molecules.

        Each inner list is for one reactant molecule.

        Args:
            zero_based: whether to convert the atom map number to zero based.
                If `True`, all atom map numbers will subtract their minimum value.

        """
        return self._get_atom_map_number(self.reactants, zero_based)

    def get_products_atom_map_number(self, zero_based=False) -> List[List[int]]:
        """
        Get the atom map number of the product molecules.

        Each inner list is for one product molecule.

        Args:
            zero_based: whether to convert the atom map number to zero based.
                If `True`, all atom map numbers will subtract their minimum value.

        """
        return self._get_atom_map_number(self.products, zero_based)

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

    def get_reactants_bond_map_number(self) -> List[List[int]]:
        """
        Get the bond map number of the reactant molecules.

        Each inner list is for one reactant molecule. The bonds are divided into two
        categories:
        1) not altered bonds: bonds exist in both the reactants and the products.
        2) lost bonds: bonds in the reactants but not in the products.

        For bonds that are not altered, a unique map number (starting from 0) is created
        for each bond. Since lost bonds do not exist in the products, there is no
        meaningful bond map number. So, we set their bond map number to `None`.
        """
        return self._get_bond_map_number(mode="reactants")

    def get_products_bond_map_number(self) -> List[List[int]]:
        """
        Get the bond map number of the product molecules.

        Each inner list is for one product molecule. The bonds are divided into two
        categories:
        1) not altered bonds: bonds exist in both the reactants and the products.
        2) added bonds: bonds not in the reactants but in the products.

        For bonds that are not altered, a unique map number (starting from 0) is created
        for each bond. Since lost bonds do not exist in the products, there is no
        meaningful bond map number. So, we set their bond map number to `None`.
        """
        return self._get_bond_map_number(mode="products")

    def _get_bond_map_number(self, mode="reactants") -> List[List[int]]:
        """
        If  `mode=="reactants"`, this generate the bond map for the reactant molecules;
        If  `mode=="products"`, this generate the bond map for the product molecules.

        """
        reactants_bonds = self._get_bonds(self.reactants)
        products_bonds = self._get_bonds(self.products)

        reactant_bonds_set = set(itertools.chain.from_iterable(reactants_bonds))
        product_bonds_set = set(itertools.chain.from_iterable(products_bonds))
        not_altered_bonds = reactant_bonds_set & product_bonds_set

        # the unique bond map number is generated as the index of the sorted array
        sorted_bonds = sorted(not_altered_bonds)

        if mode == "reactants":
            target_bonds = reactants_bonds
        elif mode == "products":
            target_bonds = products_bonds
        else:
            raise ValueError("not supported mode")

        bond_map_number = []  # map number for all molecules
        for bonds in target_bonds:
            number = []  # map number for a one molecule
            for b in bonds:
                if b in not_altered_bonds:
                    number.append(sorted_bonds.index(b))
                else:
                    number.append(None)
            bond_map_number.append(number)

        return bond_map_number

    # def get_reactants_bond_map_number_tuple_index(
    #     self, zero_based: bool = False
    # ) -> Dict[Tuple[int, int], Union[int, None]]:
    #     """
    #     Create bond map number for all the bonds in the reactant molecules.
    #     Each bond is index by a tuple (atom1, atom2), where atom1 and atom 2 are the
    #     atom map number of the two atoms forming the bond.
    #
    #     The bonds are divided into two categories:
    #
    #     1) not altered bonds: bonds exist in bonds reactants and products.
    #     2) lost bonds: bonds in the reactants but not in the products.
    #
    #     The set of not altered bonds are ordered, and the index of the bond in the
    #     ordered list is the bond map number. Since lost bonds do not exist in the
    #     products, there is no meaningful bond map number. So, we set their bond map
    #     number to `None`.
    #
    #     Args:
    #         zero_based: whether to convert the bond index to zero based.
    #             If `True`, all bond indices (atom map number) are converted to zero based.
    #
    #     Returns:
    #         Each entry of the dict `(atom1, atom2): bond_map_number` gives the bond map
    #         number for one bond, where `(atom1, atom2)` is the bond index.
    #         `bond_map_number` is `None` if the bond is lost in the reaction.
    #     """
    #     reactant_bonds = set(self._get_bonds(self.reactants, zero_based))
    #     product_bonds = set(self._get_bonds(self.products, zero_based))
    #     not_altered_bonds = reactant_bonds & product_bonds
    #     lost_bonds = reactant_bonds - not_altered_bonds
    #
    #     bond_map_number = {b: i for i, b in enumerate(sorted(not_altered_bonds))}
    #     for b in lost_bonds:
    #         bond_map_number[b] = None
    #
    #     return bond_map_number
    #
    # def get_products_bond_map_number_tuple_index(
    #     self, zero_based: bool = False
    # ) -> Dict[Tuple[int, int], Union[int, None]]:
    #     """
    #     Create bond map number for all the bonds in the reactant molecules.
    #     Each bond is index by a tuple (atom1, atom2), where atom1 and atom 2 are the
    #     atom map number of the two atoms forming the bond.
    #
    #     The bonds are divided into two categories:
    #
    #     1) not altered bonds: bonds exist in bonds reactants and products.
    #     2) added bonds: bonds not in the reactants but in the products.
    #
    #     The set of not altered bonds are ordered, and the index of the bond in the
    #     ordered list is the bond map number. Since added bonds do not exist in the
    #     reactant, there is no meaningful bond map number. So, we set their bond map
    #     number to `None`.
    #
    #     Args:
    #         zero_based: whether to convert the bond index to zero based.
    #             If `True`, all bond indices (atom map number) are converted to zero based.
    #
    #     Returns:
    #         Each entry of the dict `(atom1, atom2): bond_map_number` gives the bond map
    #         number for one bond, where `(atom1, atom2)` is the bond index.
    #         `bond_map_number` is `None` if the bond is created in the reaction.
    #     """
    #     reactant_bonds = set(self._get_bonds(self.reactants, zero_based))
    #     product_bonds = set(self._get_bonds(self.products, zero_based))
    #     not_altered_bonds = reactant_bonds & product_bonds
    #     added_bonds = product_bonds - not_altered_bonds
    #
    #     bond_map_number = {b: i for i, b in enumerate(sorted(not_altered_bonds))}
    #     for b in added_bonds:
    #         bond_map_number[b] = None
    #
    #     return bond_map_number

    # def get_changed_bonds(
    #     self, zero_based: bool = False
    # ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    #     """
    #     Get changed bonds in the reaction, i.e. lost bonds and added bonds.
    #
    #     Args:
    #         zero_based: whether to convert the bond index to zero based.
    #             If `True`, all bond indices (atom map number) are converted to zero based.
    #
    #     Returns:
    #         lost_bonds: [(atom1, atom2)], each tuple denote a lost bond. Note the atom
    #         index is  the atom map number.
    #         added_bonds: [(atom1, atom2)], each tuple denote an added bond. Note the atom
    #         index is the atom map number.
    #     """
    #     reactant_bonds = set(self._get_bonds(self.reactants, zero_based))
    #     product_bonds = set(self._get_bonds(self.products, zero_based))
    #     not_altered_bonds = reactant_bonds & product_bonds
    #     lost_bonds = list(reactant_bonds - not_altered_bonds)
    #     added_bonds = list(product_bonds - not_altered_bonds)
    #
    #     return lost_bonds, added_bonds

    @staticmethod
    def _get_bonds(
        molecules: List[Molecule], zero_based: bool = False
    ) -> List[List[Tuple[int, int]]]:
        """
        Get all the bonds in the molecules. Each bond is index as (atom1, atom2),
        where atom1 and atom2 are the atom map number for that atom.

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

        if zero_based:
            minimum = int(np.min(np.concatenate(all_bonds)))
            all_bonds = [
                [(b[0] - minimum, b[1] - minimum) for b in bonds] for bonds in all_bonds
            ]

        return all_bonds

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


def smiles_to_reaction(smiles: str, id: Optional[Union[int, str]] = None):
    """
    Convert a reaction given in smiles to :class:`Reaction`.

    Args:
        smiles: a smiles representation of a reaction, where the reactants, reagents,
            and products are separated by `>`. For example:
            '[C:1](=[O:2])-[OD1].[N!H0:3]>>[C:1](=[O:2])[N:3]'

        id: identifier of the reaction.
    """

    reactants, reagents, products = smiles.split(">")

    rcts = [Molecule.from_smiles(s) for s in reactants.split(".")]
    if reagents != "":
        rgts = [Molecule.from_smiles(s) for s in reagents.split(".")]
    else:
        rgts = []
    prdts = [Molecule.from_smiles(s) for s in products.split(".")]

    reaction = Reaction(rcts, prdts, rgts, sanity_check=False, id=id)

    return reaction


class ReactionError(Exception):
    def __init__(self, msg=None):
        super(ReactionError, self).__init__(msg)
        self.msg = msg
