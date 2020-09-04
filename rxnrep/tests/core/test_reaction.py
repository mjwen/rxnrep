from rxnrep.core.molecule import Molecule
from rxnrep.core.reaction import Reaction, ReactionSanityCheckError


def create_reaction(add_H=False):
    """
    Create a reaction: CH3CH2+ + CH3CH2- --> CH3+ + CH3CH2CH2-
    """
    m1 = Molecule.from_smiles("C[CH2+]")
    m2 = Molecule.from_smiles("C[CH2-]")
    m3 = Molecule.from_smiles("[CH3+]")
    m4 = Molecule.from_smiles("CC[CH2-]")
    mols = [m1, m2, m3, m4]

    if add_H:
        for m in mols:
            m.add_H()

    rxn = Reaction(
        reactants=[m1, m2],
        products=[m3, m4],
        id="CH3CH2+ + CH3CH2- --> CH3+ + CH3CH2CH2-",
    )

    return rxn, mols


class TestReaction:
    @staticmethod
    def assert_reaction_property(add_H=False):
        rxn, mols = create_reaction(add_H)
        assert rxn.reactants == [mols[0], mols[1]]
        assert rxn.products == [mols[2], mols[3]]
        assert rxn.id == "CH3CH2+ + CH3CH2- --> CH3+ + CH3CH2CH2-"

        # let it fail charge check
        mols[0].charge = 0
        try:
            rxn.check_charge()
        except ReactionSanityCheckError as e:
            assert "check_charge" in str(e)

        # let it fail composition check
        if add_H:
            mols[0].remove_H()
        else:
            mols[0].add_H()
        try:
            rxn.check_composition()
        except ReactionSanityCheckError as e:
            assert "check_composition" in str(e)

    def test_without_H(self):
        self.assert_reaction_property(add_H=False)

    def test_with_H(self):
        self.assert_reaction_property(add_H=True)
