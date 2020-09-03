from rxnrep.core.molecule import Molecule


def create_smiles_molecule(smiles="N([H])[CH][CH+]"):
    """
    Create a molecule CH3-CH-CH. Atoms in bracket [] are explicit Hs.

              .     .
    [H]---N---C---(C+)--[H]
          |   |
          H   [H]
    """

    return Molecule.from_smiles(smiles)


class TestMolecule:
    def test_mol_property_without_H(self):
        m = create_smiles_molecule()

        assert m.id == "N([H])[CH][CH+]"
        assert m.num_atoms == 3
        assert m.num_bonds == 2
        assert m.bonds == [(0, 1), (1, 2)]
        assert m.formal_charge == m.charge == 1
        assert m.species == ["N", "C", "C"]
        assert m.composition_dict == {"N": 1, "C": 2}
        assert m.formula == "C2N1"
        assert m.generate_coords().shape == (3, 3)
        assert m.coords.shape == (3, 3)
        assert m.optimize_coords().shape == (3, 3)

    def test_mol_property_with_H(self):
        m = create_smiles_molecule()
        m.add_H(explicit_only=True)

        assert m.id == "N([H])[CH][CH+]"
        assert m.num_atoms == 6
        assert m.num_bonds == 5
        assert m.bonds == [
            (0, 1),
            (1, 2),
            (0, 3),
            (1, 4),
            (2, 5),
        ]
        assert m.formal_charge == m.charge == 1
        assert m.species == ["N", "C", "C", "H", "H", "H"]
        assert m.composition_dict == {"N": 1, "C": 2, "H": 3}
        assert m.formula == "C2H3N1"
        assert m.generate_coords().shape == (6, 3)
        assert m.coords.shape == (6, 3)
        assert m.optimize_coords().shape == (6, 3)
