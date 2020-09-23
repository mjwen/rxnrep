from rxnrep.core.molecule import Molecule


def create_smiles_molecule():
    """
    Create a molecule CH3-CH-CH. Atoms in bracket [] are explicit Hs.

              .     .
    [H]---N---C---(C+)--[H]
          |   |
          H   [H]
    """
    smiles = "N([H])[CH][CH+]"
    return Molecule.from_smiles(smiles)


def create_sdf_molecule(option=1):
    """
    Create the same molecule in `create_smiles_molecule`, but from sdf.

    Option 1: smiles without H;
    Option 2: smiles with explicity H;
    Option 3: smiles with both explicity and implicit H

    It's seem that the N, C, C atoms are all specified the same in all three cases in
    terms of radicals (RAD), valence (VAL), and charges (CHG), making sure they are
    the same molecule.

    Option 1:

    N([H])[CH][CH+]
         RDKit          2D

      0  0  0  0  0  0  0  0  0  0999 V3000
    M  V30 BEGIN CTAB
    M  V30 COUNTS 3 2 0 0 0
    M  V30 BEGIN ATOM
    M  V30 1 N 2.59808 -6.66134e-16 0 0
    M  V30 2 C 1.29904 0.75 0 0 RAD=2 VAL=3
    M  V30 3 C 0 0 0 0 CHG=1 RAD=2 VAL=2
    M  V30 END ATOM
    M  V30 BEGIN BOND
    M  V30 1 1 1 2
    M  V30 2 1 2 3
    M  V30 END BOND
    M  V30 END CTAB
    M  END

    Option 2:

    N([H])[CH][CH+]
         RDKit          2D

      0  0  0  0  0  0  0  0  0  0999 V3000
    M  V30 BEGIN CTAB
    M  V30 COUNTS 6 5 0 0 0
    M  V30 BEGIN ATOM
    M  V30 1 N 2.59808 -6.66134e-16 0 0
    M  V30 2 C 1.29904 0.75 0 0 RAD=2 VAL=3
    M  V30 3 C 0 0 0 0 CHG=1 RAD=2 VAL=2
    M  V30 4 H 3.89711 0.75 0 0
    M  V30 5 H 1.29904 2.25 0 0
    M  V30 6 H -1.29904 0.75 0 0
    M  V30 END ATOM
    M  V30 BEGIN BOND
    M  V30 1 1 1 2
    M  V30 2 1 2 3
    M  V30 3 1 1 4
    M  V30 4 1 2 5
    M  V30 5 1 3 6
    M  V30 END BOND
    M  V30 END CTAB
    M  END

    N([H])[CH][CH+]
         RDKit          2D

      0  0  0  0  0  0  0  0  0  0999 V3000
    M  V30 BEGIN CTAB
    M  V30 COUNTS 7 6 0 0 0
    M  V30 BEGIN ATOM
    M  V30 1 N 2.59808 -6.66134e-16 0 0
    M  V30 2 C 1.29904 0.75 0 0 RAD=2 VAL=3
    M  V30 3 C 0 0 0 0 CHG=1 RAD=2 VAL=2
    M  V30 4 H 3.89711 0.75 0 0
    M  V30 5 H 2.59808 -1.5 0 0
    M  V30 6 H 1.29904 2.25 0 0
    M  V30 7 H -1.29904 0.75 0 0
    M  V30 END ATOM
    M  V30 BEGIN BOND
    M  V30 1 1 1 2
    M  V30 2 1 2 3
    M  V30 3 1 1 4
    M  V30 4 1 1 5
    M  V30 5 1 2 6
    M  V30 6 1 3 7
    M  V30 END BOND
    M  V30 END CTAB
    M  END
    """
    m = create_smiles_molecule()

    if option == 1:
        pass
    elif option == 2:
        m.add_H(explicit_only=True)
    elif option == 3:
        m.add_H(explicit_only=False)
    else:
        raise Exception
    sdf = m.to_sdf()

    return Molecule.from_sdf(sdf)


class TestMolecule:
    @staticmethod
    def assert_mol_property_without_H(m, id):

        assert m.id == id
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

    @staticmethod
    def assert_mol_property_with_explicit_H(m, id):

        assert m.id == id
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

    @staticmethod
    def assert_mol_property_with_all_H(m, id):

        assert m.id == id
        assert m.num_atoms == 7
        assert m.num_bonds == 6
        assert m.bonds == [
            (0, 1),
            (1, 2),
            (0, 3),
            (0, 4),
            (1, 5),
            (2, 6),
        ]
        assert m.formal_charge == m.charge == 1
        assert m.species == ["N", "C", "C", "H", "H", "H", "H"]
        assert m.composition_dict == {"N": 1, "C": 2, "H": 4}
        assert m.formula == "C2H4N1"
        assert m.generate_coords().shape == (7, 3)
        assert m.coords.shape == (7, 3)
        assert m.optimize_coords().shape == (7, 3)

    def test_smiles(self):
        m = create_smiles_molecule()
        self.assert_mol_property_without_H(m, "N([H])[CH][CH+]")

        m = create_smiles_molecule()
        m.add_H(explicit_only=True)
        self.assert_mol_property_with_explicit_H(m, "N([H])[CH][CH+]")

        m = create_smiles_molecule()
        m.add_H(explicit_only=False)
        self.assert_mol_property_with_all_H(m, "N([H])[CH][CH+]")

    def test_sdf(self):
        m = create_sdf_molecule(option=1)
        self.assert_mol_property_without_H(m, None)

        m = create_sdf_molecule(option=2)
        self.assert_mol_property_with_explicit_H(m, None)

        m = create_sdf_molecule(option=3)
        self.assert_mol_property_with_all_H(m, None)

    def test_atom_map_number(self):
        m = create_smiles_molecule()
        assert m.get_atom_map_number() == [None, None, None]

        m.set_atom_map_number({0: 2, 1: 1, 2: 3})
        assert m.get_atom_map_number() == [2, 1, 3]

        smiles = "[N:2]([H])[CH:1][CH+:3]"
        m = Molecule.from_smiles(smiles)
        assert m.get_atom_map_number() == [2, 1, 3]
