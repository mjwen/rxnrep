from rxnrep.core.molecule import Molecule
from rxnrep.core.reaction import Reaction, ReactionError, smiles_to_reaction


def create_reaction(add_H=False):
    """
    Create a reaction: CH3CH2+ + CH3CH2 --> CH3 + CH3CH2CH2+
    """
    m1 = Molecule.from_smiles("[CH3:3][CH2+:1]")
    m2 = Molecule.from_smiles("[CH3:2][CH2:4]")
    m3 = Molecule.from_smiles("[CH3:3]")
    m4 = Molecule.from_smiles("[CH3:1][CH2:2][CH2+:4]")
    mols = [m1, m2, m3, m4]

    if add_H:
        for m in mols:
            m.add_H()

    rxn = Reaction(
        reactants=[m1, m2], products=[m3, m4], id="some text as id", sanity_check=False
    )

    return rxn, mols


class TestReaction:
    @staticmethod
    def assert_reaction_property(add_H=False):
        rxn, mols = create_reaction(add_H)
        assert rxn.reactants == [mols[0], mols[1]]
        assert rxn.products == [mols[2], mols[3]]
        assert rxn.id == "some text as id"

        # let it fail charge check
        mols[0].charge = 0
        try:
            rxn.check_charge()
        except ReactionError as e:
            assert "check_charge" in str(e)

        # let it fail composition check
        if add_H:
            mols[0].remove_H()
        else:
            mols[0].add_H()
        try:
            rxn.check_composition()
        except ReactionError as e:
            assert "check_composition" in str(e)

    def test_without_H(self):
        self.assert_reaction_property(add_H=False)

    def test_with_H(self):
        self.assert_reaction_property(add_H=True)

    def test_species(self):
        smi = "[CH3:3][NH2:1].[PH4:2][O:4]>>[CH3:3].[NH1:1][PH3:2][O:4]"
        rxn = smiles_to_reaction(smi, smi, sanity_check=False)
        assert rxn.species == ["N", "P", "C", "O"]

    def test_get_bonds(self):
        rxn, _ = create_reaction()

        # reactants
        bonds = rxn.get_reactants_bonds(zero_based=False)
        assert bonds == [[(1, 3)], [(2, 4)]]
        bonds = rxn.get_reactants_bonds(zero_based=True)
        assert bonds == [[(0, 2)], [(1, 3)]]

        # products
        bonds = rxn.get_products_bonds(zero_based=False)
        assert bonds == [[], [(1, 2), (2, 4)]]
        bonds = rxn.get_products_bonds(zero_based=True)
        assert bonds == [[], [(0, 1), (1, 3)]]

    def test_get_atom_map_number(self):
        rxn, _ = create_reaction()

        # reactants
        map_number = rxn.get_reactants_atom_map_number(zero_based=False)
        assert map_number == [[3, 1], [2, 4]]
        map_number = rxn.get_reactants_atom_map_number(zero_based=True)
        assert map_number == [[2, 0], [1, 3]]

        # products
        map_number = rxn.get_products_atom_map_number(zero_based=False)
        assert map_number == [[3], [1, 2, 4]]
        map_number = rxn.get_products_atom_map_number(zero_based=True)
        assert map_number == [[2], [0, 1, 3]]

    def test_get_bond_map_number(self):
        rxn, _ = create_reaction()

        # reactants
        bond_map_number = rxn.get_reactants_bond_map_number(for_changed=False)
        assert bond_map_number == [[None], [0]]
        bond_map_number = rxn.get_reactants_bond_map_number(for_changed=True)
        assert bond_map_number == [[1], [0]]

        # products
        bond_map_number = rxn.get_products_bond_map_number(for_changed=False)
        assert bond_map_number == [[], [None, 0]]

    def test_check_atom_map_number(self):
        rxn, mols = create_reaction(add_H=False)
        rxn.check_atom_map_number()

        # let atom have no atom map number
        mols[0].set_atom_map_number({0: None})
        try:
            rxn.check_atom_map_number()
        except ReactionError as e:
            assert "check_atom_map_number" in str(e)
        mols[0].set_atom_map_number({0: 3})  # set back

        # let atom have the same map number
        mols[0].set_atom_map_number({0: 1})
        try:
            rxn.check_atom_map_number()
        except ReactionError as e:
            assert "check_atom_map_number" in str(e)
        mols[0].set_atom_map_number({0: 3})  # set back

        # let reactants and products have different atom map number
        mols[0].set_atom_map_number({0: 5})
        try:
            rxn.check_atom_map_number()
        except ReactionError as e:
            assert "check_atom_map_number" in str(e)
        mols[0].set_atom_map_number({0: 3})  # set back
