import logging
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
        atom_mapping: atom mapping between the reactants and the products. Each dict is
            the atom mapping for a reactant. (int, (int, int)) means
            (reactant_atom_index: (product_index, product_atom_index)).
        sanity_check: check the correctness of the reactions, e.g. mass conservation,
            charge conservation...
        id: a string identifier of the reaction
    """

    def __init__(
        self,
        reactants: List[Molecule],
        products: List[Molecule],
        reagents: Optional[List[Molecule]] = None,
        atom_mapping: Optional[List[Dict[int, Tuple[int, int]]]] = None,
        sanity_check: bool = True,
        id: Optional[str] = None,
    ):

        self._reactants = reactants
        self._products = products
        self._reagents = reagents
        self._atom_mapping = atom_mapping
        self._id = id

        if sanity_check:
            self.sanity_check()

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
    def id(self) -> Union[str, None]:
        """
        Returns a string identification (name) of the reaction.
        """
        return self._id

    def get_atom_mapping(self) -> List[Dict[int, Tuple[int, int]]]:
        """
        Returns the atom mapping between reactants and products.
        """
        return self._atom_mapping

    def set_atom_mapping(self, mapping: List[Dict[int, Tuple[int, int]]]):
        """
        Set the atom mapping between reactants and products.

        Args:
            mapping: each dict is the atom mapping for a reactant. (int, (int, int))
                means (reactant_atom_index: (product_index, product_atom_index)).
        """
        self._atom_mapping = mapping
        self.check_atom_mapping()

    def sanity_check(self):
        """
        Check the correctness of the reaction.
        """
        self.check_composition()
        self.check_charge()
        if self._atom_mapping is not None:
            self.check_atom_mapping()

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

            raise ReactionSanityCheckError(
                f"Failed `check_composition()` for reaction {self.id}. "
                f"Reactants composition is {rc}, while products composition is {pc}."
            )

    def check_charge(self):
        """
        Check that charge is balanced between the reactants and products.
        """
        reactants_charge = sum([m.charge for m in self._reactants])
        products_charge = sum([m.charge for m in self._products])
        if reactants_charge != products_charge:
            raise ReactionSanityCheckError(
                f"Failed `check_charge()` for reaction {self.id}. "
                f"The sum of reactant charges ({reactants_charge}) "
                f"does not equal the sum of product charges ({products_charge})."
            )

    def check_atom_mapping(self):
        """
        Check the correctness of atom mapping: every reactant and product atom is
        mapped, and each should have one and only one map.
        """
        reactants = defaultdict(list)
        products = defaultdict(list)
        for i, mp in enumerate(self._atom_mapping):
            for r_atom, (p_molecule, p_atom) in mp.items():
                reactants[i].append(r_atom)
                products[p_molecule].append(p_atom)

        # check every reactant is mapped
        for i, m in enumerate(self._reactants):
            if list(range(m.num_atoms)) != sorted(reactants[i]):
                raise ReactionSanityCheckError(
                    f"Failed `check_atom_mapping()` for reaction {self.id}. "
                    f"Reactant {i} has {m.num_atoms} atoms; but mapped atoms for it "
                    f"is {reactants[i]}."
                )

        # check every product is mapped
        for i, m in enumerate(self._products):
            if list(range(m.num_atoms)) != sorted(products[i]):
                raise ReactionSanityCheckError(
                    f"Failed `check_atom_mapping()` for reaction {self.id}. "
                    f"Product {i} has {m.num_atoms} atoms; but mapped atoms for it "
                    f"is {reactants[i]}."
                )

    #
    # def get_broken_bond(self):
    #     """
    #     Returns:
    #         tuple: sorted index of broken bond (a 2-tuple of atom index)
    #     """
    #     raise NotImplementedError
    #
    # def bond_mapping_by_int_index(self):
    #     r"""
    #     Find the bond mapping between products and reactant, using a single index (the
    #     index of bond in MoleculeWrapper.bonds) to denote the bond.
    #
    #     For example, suppose we have reactant
    #
    #           C 0
    #        0 / \ 1
    #         /___\  3   4
    #        O  2  N---O---H
    #        1     2   3  4
    #
    #     products
    #           C 0
    #        1 / \ 0
    #         /___\
    #        O  2  N
    #        1     2
    #     and (note the index of H changes it 0)
    #           0
    #         O---H
    #         0   1
    #     The function gives the bond mapping:
    #     [{0:1, 1:0, 2:2}, {0:4}]
    #
    #
    #     The mapping is done by finding correspondence between atoms indices of reactant
    #     and products.
    #
    #     Returns:
    #         list: each element is a dict mapping the bonds from product to reactant
    #     """
    #
    #     # mapping between tuple index and integer index for the same bond
    #     reactant_mapping = {
    #         bond: ordering
    #         for ordering, (bond, _) in enumerate(self._reactants[0].bonds.items())
    #     }
    #
    #     bond_mapping = []
    #
    #     for p, amp in zip(self._products, self.atom_mapping):
    #         bmp = dict()
    #
    #         for p_ordering, (bond, _) in enumerate(p.bonds.items()):
    #
    #             # atom mapping between product and reactant of the bond
    #             bond_amp = [amp[i] for i in bond]
    #
    #             r_ordering = reactant_mapping[tuple(sorted(bond_amp))]
    #             bmp[p_ordering] = r_ordering
    #         bond_mapping.append(bmp)
    #
    #     return bond_mapping
    #
    # def bond_mapping_by_tuple_index(self):
    #     r"""
    #     Find the bond mapping between products and reactant, using a tuple index (atom
    #     index) to denote the bond.
    #
    #     For example, suppose we have reactant
    #
    #           C 0
    #        0 / \ 1
    #         /___\  3   4
    #        O  2  N---O---H
    #        1     2   3  4
    #
    #     products
    #           C 0
    #        1 / \ 0
    #         /___\
    #        O  2  N
    #        2     1
    #     and (note the index of H changes it 0)
    #           0
    #         O---H
    #         0   1
    #     The function will give the bond mapping:
    #     [{(0,1):(0,2), (0,2):(0,1), (1,2):(1,2)}, {(0,1):(3,4)}]
    #
    #
    #     The mapping is done by finding correspondence between atoms indices of reactant
    #     and products.
    #
    #     Returns:
    #         list: each element is a dict mapping the bonds from a product to reactant
    #     """
    #
    #     bond_mapping = []
    #
    #     for p, amp in zip(self._products, self.atom_mapping):
    #         bmp = dict()
    #
    #         for b_product in p.bonds:
    #
    #             # atom mapping between product and reactant of the bond
    #             i, j = b_product
    #
    #             b_reactant = tuple(sorted([amp[i], amp[j]]))
    #             bmp[b_product] = b_reactant
    #         bond_mapping.append(bmp)
    #
    #     return bond_mapping
    #
    # def bond_mapping_by_sdf_int_index(self):
    #     """
    #     Bond mapping between products SDF bonds (integer index) and reactant SDF bonds
    #     (integer index).
    #
    #     Unlike the atom mapping (where atom index in graph and sdf are the same),
    #     the ordering of bond may change when sdf file are written. So we need this
    #     mapping to ensure the correct ordering between products bonds and reactant bonds.
    #
    #     We do the below to get a mapping between product sdf int index and reactant
    #     sdf int index:
    #
    #     product sdf int index
    #     --> product sdf tuple index
    #     --> product graph tuple index
    #     --> reactant graph tuple index
    #     --> reactant sdf tuple index
    #     --> reactant sdf int index
    #
    #
    #     Returns:
    #         list (dict): each dict is the mapping for one product, from sdf bond index
    #             of product to sdf bond index of reactant
    #     """
    #
    #     reactant = self._reactants[0]
    #
    #     # reactant sdf bond index (tuple) to sdf bond index (integer)
    #     reactant_index_tuple2int = {
    #         b: i for i, b in enumerate(reactant.get_sdf_bond_indices(zero_based=True))
    #     }
    #
    #     # bond mapping between product sdf and reactant sdf
    #     bond_mapping = []
    #     product_to_reactant_mapping = self.bond_mapping_by_tuple_index()
    #     for p, p2r in zip(self._products, product_to_reactant_mapping):
    #
    #         mp = {}
    #         # product sdf bond index (list of tuple)
    #         psb = p.get_sdf_bond_indices(zero_based=True)
    #
    #         # ib: product sdf bond index (int)
    #         # b: product graph bond index (tuple)
    #         for ib, b in enumerate(psb):
    #
    #             # reactant graph bond index (tuple)
    #             rsbt = p2r[b]
    #
    #             # reactant sdf bond index (int)
    #             rsbi = reactant_index_tuple2int[rsbt]
    #
    #             # product sdf bond index (int) to reactant sdf bond index (int)
    #             mp[ib] = rsbi
    #
    #         # list of dict, each dict for one product
    #         bond_mapping.append(mp)
    #
    #     return bond_mapping


class ReactionSanityCheckError(Exception):
    def __init__(self, msg=None):
        self.msg = msg
        super(ReactionSanityCheckError, self).__init__(msg)

    def __repr__(self):
        return self.msg
