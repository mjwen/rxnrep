import copy
import itertools
import logging
from collections import OrderedDict, defaultdict
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from sklearn.utils import class_weight

from rxnrep.core.molecule import Molecule, find_functional_group
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

        self._atom_distance_to_reaction_center = None
        self._bond_distance_to_reaction_center = None
        self._reaction_center_atom_functional_group = None

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
    def num_atoms(self) -> int:
        """
        Total number of atoms.
        """
        return sum([m.num_atoms for m in self.reactants])

    @property
    def num_reactants_bonds(self) -> int:
        """
        Total number of bonds in all reactants.
        """
        return len(self.unchanged_bonds) + len(self.lost_bonds)

    @property
    def num_products_bonds(self) -> int:
        """
        Total number of bonds in all products.
        """
        return len(self.unchanged_bonds) + len(self.added_bonds)

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

    @property
    def atom_distance_to_reaction_center(self) -> List[int]:
        """
        Hop distance of atoms to reaction center.

        The atoms are ordered according to atom map number.
        """
        if self._atom_distance_to_reaction_center is None:

            atoms_in_reaction_center = set(
                [
                    i
                    for i in itertools.chain.from_iterable(
                        self.lost_bonds + self.added_bonds
                    )
                ]
            )

            all_bonds = self.unchanged_bonds + self.lost_bonds + self.added_bonds

            # row of distances: atoms in the center;
            # column of distances: distance to other atoms
            VAL = 10000000000
            distances = VAL * np.ones((self.num_atoms, self.num_atoms), dtype=np.int32)

            # distance from center atoms to other atoms
            nx_graph = nx.Graph(incoming_graph_data=all_bonds)
            for center_atom in atoms_in_reaction_center:
                dists = nx.single_source_shortest_path_length(nx_graph, center_atom)
                for atom, d in dists.items():
                    distances[center_atom][atom] = d

            distances = np.min(distances, axis=0).tolist()

            if VAL in distances:
                atom = distances.index(VAL)
                raise RuntimeError(
                    f"Cannot find path to reaction center for atom {atom}, this should not "
                    "happen. The reaction probably has atoms not connected to others in "
                    "both the reactants and the products. Please remove these atoms."
                    f"Bad reaction is: {self.id}"
                )

            self._atom_distance_to_reaction_center = distances

        return self._atom_distance_to_reaction_center

    @property
    def bond_distance_to_reaction_center(self) -> List[int]:
        """
        Hop distance of bonds to reaction center.

        Bonds are ordered according to bond map number.

        In `combine_graphs()`, the bond node in the graph are reordered according to bond
        map number. In `create_reaction_graph()`, the unchanged bonds will have bond
        node number 0, 1, ... N_unchanged-1, the lost bonds in the reactants will have
        bond node number N_unchanged, ... N-1, where N is the number of bonds in the
        reactants, and the added bonds will have bond node number N, ... N+N_added-1.
        We shifted the indices of the added bonds right by `the number of lost bonds`
        to make a graph containing all bonds. Here we do the same shift for added bonds.
        """

        if self._bond_distance_to_reaction_center is None:

            atom_distances = self.atom_distance_to_reaction_center

            reactants_bond_map_number = self.get_reactants_bond_map_number(
                for_changed=True, as_dict=True
            )
            products_bond_map_number = self.get_products_bond_map_number(
                for_changed=True, as_dict=True
            )
            reactants_bond_map_number = {
                k: v for d in reactants_bond_map_number for k, v in d.items()
            }
            products_bond_map_number = {
                k: v for d in products_bond_map_number for k, v in d.items()
            }

            unchanged_bonds = self.unchanged_bonds
            lost_bonds = self.lost_bonds
            added_bonds = self.added_bonds
            num_lost_bonds = len(lost_bonds)
            num_bonds = len(unchanged_bonds + lost_bonds + added_bonds)

            distances = [None] * num_bonds

            for bond in lost_bonds:
                idx = reactants_bond_map_number[bond]
                distances[idx] = 0

            for bond in added_bonds:
                idx = products_bond_map_number[bond] + num_lost_bonds
                distances[idx] = 0

            for bond in unchanged_bonds:
                atom1, atom2 = bond
                atom1_dist = atom_distances[atom1]
                atom2_dist = atom_distances[atom2]

                if atom1_dist == atom2_dist:
                    dist = atom1_dist + 1
                else:
                    dist = max(atom1_dist, atom2_dist)

                idx = reactants_bond_map_number[bond]
                distances[idx] = dist

            if None in distances:
                raise RuntimeError(
                    "Some bond has not hop distance, this should not happen. "
                    "Bad reaction is: {self.id}"
                )

            self._bond_distance_to_reaction_center = distances

        return self._bond_distance_to_reaction_center

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

    def get_reaction_center_atom_functional_group(
        self, func_groups: Union[Path, List], include_center_atoms: bool = True
    ) -> List[int]:
        """
        The functional groups associated with atoms in reaction center.

        For each molecule, get largest functional group associated with atoms in
        reaction center and take the union of functional group of all molecules.

        Args:
            func_groups: if a Path, should be a Path to a tsv file containing the SMARTS
                of the functional group. Or it could be a list of rdkit mols
                created by MolFromSmarts.
            include_center_atoms: whether to include center atoms in the returned
                functional group atoms.

        Returns:
            func_atom_indexes: functional group atoms associated with
                atoms in reaction center. The returned atoms are nide
        """

        if self._reaction_center_atom_functional_group is None:

            reactants = [m.rdkit_mol for m in self.reactants]
            products = [m.rdkit_mol for m in self.products]
            dist = self.atom_distance_to_reaction_center

            rct_atom_map = self.get_reactants_atom_map_number()
            prdt_atom_mapping = self.get_products_atom_map_number()

            all_fg_atoms = set()
            for m, atom_map in zip(
                reactants + products, rct_atom_map + prdt_atom_mapping
            ):
                center_atom = [i for i, m in enumerate(atom_map) if dist[m] == 0]
                fg_atoms = find_functional_group(m, center_atom, func_groups)

                # change atom index to map number index
                fg_atoms = [atom_map[i] for i in fg_atoms]

                all_fg_atoms.update(fg_atoms)
                if include_center_atoms:
                    all_fg_atoms.update([atom_map[i] for i in center_atom])

            self._reaction_center_atom_functional_group = list(all_fg_atoms)

        return self._reaction_center_atom_functional_group

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

    def draw2(
        self, filename: Path = None, image_size=(800, 300), format="svg", font_scale=1.5
    ):
        """
        The returned image can be viewed in Jupyter with display(SVG(image)).

        Args:
            filename:
            font_scale:
            image_size:
            format:

        Returns:
        """

        rxn = AllChem.ReactionFromSmarts(str(self), useSmiles=True)

        if format == "png":
            d2d = rdMolDraw2D.MolDraw2DCairo(*image_size)
        elif format == "svg":
            d2d = rdMolDraw2D.MolDraw2DSVG(*image_size)
        else:
            supported = ["png", "svg"]
            raise ValueError(f"Supported format are {supported}; got {format}")

        # d2d.SetFontSize(font_scale * d2d.FontSize())

        d2d.DrawReaction(rxn)
        d2d.FinishDrawing()

        img = d2d.GetDrawingText()

        if filename is not None:
            with open(filename, "wb") as f:
                f.write(img)

        return img

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
    for atom in range(reaction.num_atoms):

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
    A bond connected to the reaction center via 1 other bond has a hop distance of `2`;
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

    reactants_bond_map_number = reaction.get_reactants_bond_map_number(
        for_changed=True, as_dict=True
    )
    products_bond_map_number = reaction.get_products_bond_map_number(
        for_changed=True, as_dict=True
    )
    reactants_bond_map_number = {
        k: v for d in reactants_bond_map_number for k, v in d.items()
    }
    products_bond_map_number = {
        k: v for d in products_bond_map_number for k, v in d.items()
    }

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
        idx = reactants_bond_map_number[bond]
        hop_distances[idx] = 0

    for bond in added_bonds:
        idx = products_bond_map_number[bond] + num_lost_bonds
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

        idx = reactants_bond_map_number[bond]
        hop_distances[idx] = dist

    assert None not in hop_distances, (
        "Some bond has not hop distance, this should not happen. Bad reaction is: :"
        f"{reaction.id}"
    )

    return hop_distances


def get_atom_bond_hop_dist_class_weight(
    labels: List[Dict[str, torch.Tensor]],
    max_hop_distance: int,
    only_break_bond: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Create class weight to be used in cross entropy losses.

    Args:
        labels: a list of dict that contains `atom_hop_dist` and `bond_hop_dist` labels.
            See `get_atom_distance_to_reaction_center()` and
            `get_bond_distance_to_reaction_center` in grapher.py for how these labels
            are generated and what the mean.
        max_hop_distance: max hop distance used to generate the labels.
        only_break_bond: in all the reactions, there is only bond breaking, but not
            bond forming. If there is no bond forming, allowed number of classes for atom
            and bond hop distances are both different from reactions with bond forming.

    Returns:
        {name: weight}: class weight for atom/bond hop distance labels
    """

    if only_break_bond:
        atom_hop_num_classes = [max_hop_distance + 1]
        bond_hop_num_classes = max_hop_distance + 1
    else:
        atom_hop_num_classes = [max_hop_distance + 2, max_hop_distance + 3]
        bond_hop_num_classes = max_hop_distance + 2

    # Unique labels should be `list(range(atom_hop_num_classes))`.
    # The labels are:
    # atoms in lost bond: class 0
    # atoms in unchanged bond: class 1 to max_hop_distance
    # If there are added bonds
    # atoms in added bonds: class max_hop_distance + 1
    # If there are atoms associated with both lost and added bonds, the class label for
    # them is: max_hop_distance + 2
    all_atom_hop_labels = np.concatenate([lb["atom_hop_dist"] for lb in labels])

    unique_labels = sorted(set(all_atom_hop_labels))
    if unique_labels not in [list(range(a)) for a in atom_hop_num_classes]:
        raise RuntimeError(
            f"Unable to compute atom class weight; some classes do not have valid "
            f"labels. num_classes: {atom_hop_num_classes} unique labels: "
            f"{unique_labels}."
        )

    atom_hop_weight = class_weight.compute_class_weight(
        "balanced",
        classes=unique_labels,
        y=all_atom_hop_labels,
    )

    # Unique labels should be `list(range(bond_hop_num_classes))`.
    # The labels are:
    # lost bond: class 0
    # unchanged: class 1 to max_hop_distance
    # If there are added bonds:
    # add bonds: class max_hop_distance + 1
    all_bond_hop_labels = np.concatenate([lb["bond_hop_dist"] for lb in labels])

    unique_labels = sorted(set(all_bond_hop_labels))
    if unique_labels != list(range(bond_hop_num_classes)):
        raise RuntimeError(
            f"Unable to compute bond class weight; some classes do not have valid "
            f"labels. num_classes: {bond_hop_num_classes} unique labels: "
            f"{unique_labels}"
        )

    bond_hop_weight = class_weight.compute_class_weight(
        "balanced",
        classes=unique_labels,
        y=all_bond_hop_labels,
    )

    weight = {
        "atom_hop_dist": torch.as_tensor(atom_hop_weight, dtype=torch.float32),
        "bond_hop_dist": torch.as_tensor(bond_hop_weight, dtype=torch.float32),
    }

    return weight
