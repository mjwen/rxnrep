from rxnrep.core.utils import generate_atom_map_number_one_bond_break_reaction


def test_atom_mapping():
    smiles = "NCCCC(=O)O>>[CH2]CCC(=O)O.[NH2]"

    s1 = generate_atom_map_number_one_bond_break_reaction(smiles, add_H=False)
    s2 = generate_atom_map_number_one_bond_break_reaction(smiles, add_H=True)

    ref1 = "[NH2:1][CH2:2][CH2:3][CH2:4][C:5](=[O:6])[OH:7]>>[CH2:2][CH2:3][CH2:4][C:5](=[O:6])[OH:7].[NH2:1]"
    assert s1 == ref1

    ref2 = "[N:1]([C:2]([C:3]([C:4]([C:5](=[O:6])[O:7][H:16])([H:14])[H:15])([H:12])[H:13])([H:10])[H:11])([H:8])[H:9]>>[C:2]([C:3]([C:4]([C:5](=[O:6])[O:7][H:16])([H:14])[H:15])([H:12])[H:13])([H:10])[H:11].[N:1]([H:8])[H:9]"
    assert s2 == ref2
