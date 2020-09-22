from rxnrep.core.molecule import Molecule
from rxnrep.core.reaction import Reaction, ReactionError


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

    def test_atom_map_number(self):
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

    def test_get_changed_bonds(self):
        rxn, mols = create_reaction(add_H=False)

        lost_bonds, added_bonds = rxn.get_changed_bonds(zero_based=False)
        assert lost_bonds == [(1, 3)]
        assert added_bonds == [(1, 2)]

        lost_bonds, added_bonds = rxn.get_changed_bonds(zero_based=True)
        assert lost_bonds == [(0, 2)]
        assert added_bonds == [(0, 1)]

    def test_get_reactants_bond_map_number(self):
        rxn, mols = create_reaction(add_H=False)
        bond_map_number = rxn.get_reactants_bond_map_number(zero_based=True)
        assert bond_map_number == {(1, 3): 0, (0, 2): None}

    def test_get_products_bond_map_number(self):
        rxn, mols = create_reaction(add_H=False)
        bond_map_number = rxn.get_products_bond_map_number(zero_based=True)
        assert bond_map_number == {(1, 3): 0, (0, 1): None}
