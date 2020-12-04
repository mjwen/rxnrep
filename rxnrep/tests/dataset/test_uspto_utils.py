from rdkit import Chem

from rxnrep.dataset.uspto_utils import (
    add_nonexist_atoms_and_bonds_to_product,
    adjust_atom_map_number,
    adjust_reagents,
    canonicalize_smiles_reaction,
    canonicalize_smiles_reaction_by_adding_nonexist_atoms_and_bonds,
    edit_molecule,
    get_atom_property_as_dict,
    get_bond_change_nonexist_atoms,
    get_reaction_bond_change,
)


def get_smi_reaction():
    """
    Return an artificial smiles reaction:

    CH3CH2+ + CH3CH2 + Na+ + K+  -- Cl- --> CH3 + CH3CH2CH2+
    """
    reactant = "[CH3:1][CH2+:2].[CH3:3][CH2:4].[Na+:5].[K+]"
    reagent = "[Cl-]"
    product = "[CH3:1].[CH3:3][CH2:4][CH2+:2]"
    reaction = ">".join([reactant, reagent, product])
    return reaction


def get_reactant_reagent_and_product():
    rxn_smi = get_smi_reaction()
    reactant, reagent, product = rxn_smi.split(">")
    reactant = Chem.MolFromSmiles(reactant)
    reagent = Chem.MolFromSmiles(reagent)
    product = Chem.MolFromSmiles(product)
    return reactant, reagent, product


def test_adjust_reagents():
    rxn_smi = get_smi_reaction()
    rxn = adjust_reagents(rxn_smi)
    reactant, reagent, product = rxn.strip().split(">")
    assert reactant == "[CH3:1][CH2+:2].[CH3:3][CH2:4]"
    assert reagent == "[Cl-].[Na+].[K+]"
    assert product == "[CH3:1].[CH2+:2][CH2:4][CH3:3]"


def test_adjust_atom_map_number():
    reactant, _, product = get_reactant_reagent_and_product()
    reactant, product = adjust_atom_map_number(reactant, product)
    rct_smi = Chem.MolToSmiles(reactant)
    prdt_smi = Chem.MolToSmiles(product)
    assert rct_smi == "[CH3:1][CH2+:4].[CH3:2][CH2:3].[K+:6].[Na+:5]"
    assert prdt_smi == "[CH3:1].[CH3:2][CH2:3][CH2+:4]"


def test_get_reaction_bond_change():
    reactant, _, product = get_reactant_reagent_and_product()

    changes, _, _, _ = get_reaction_bond_change(
        reactant, product, use_mapped_atom_index=True
    )
    assert changes == {(1, 2, 0.0), (2, 4, 1.0)}

    changes, _, _, _ = get_reaction_bond_change(
        reactant, product, use_mapped_atom_index=False
    )
    assert changes == {(0, 1, 0.0), (1, 3, 1.0)}


def test_edit_molecule():
    reactant, _, product = get_reactant_reagent_and_product()
    edits = {(0, 1, 0.0), (1, 3, 1.0)}
    product_atom_props = get_atom_property_as_dict(product)
    edited_mol = edit_molecule(reactant, edits, product_atom_props)
    edited_mol = Chem.MolToSmiles(edited_mol)
    assert edited_mol == "[CH2+:2][CH2:4][CH3:3].[CH3:1].[K+].[Na+:5]"


def test_canonicalize_smiles_reaction():
    rxn_smi = get_smi_reaction()
    canonical_rxn_smi, error = canonicalize_smiles_reaction(rxn_smi)
    reactant, reagent, product = canonical_rxn_smi.split(">")

    ref_reactant = "[CH3:1][CH2+:2].[CH2:3][CH3:4]"
    ref_reagent = "[Cl-].[Na+].[K+]"
    ref_product = "[CH3:1].[CH2+:2][CH2:3][CH3:4]"

    assert error is None
    assert set(reactant.split(".")) == set(ref_reactant.split("."))
    assert set(product.split(".")) == set(ref_product.split("."))
    assert set(reagent.split(".")) == set(ref_reagent.split("."))


def test_get_bond_change_nonexist_atoms():
    reactant = Chem.MolFromSmiles("[CH3:1][CH2+:2].[CH2:3][CH3:4]")
    product = Chem.MolFromSmiles("[CH3:1].[CH2+:2][CH2:3][CH3:4]")
    bond_change = get_bond_change_nonexist_atoms(reactant, product)
    assert bond_change == set()

    reactant = Chem.MolFromSmiles("[CH3:1][CH2+:2].[CH2:3][CH3:4]")
    product = Chem.MolFromSmiles("[CH3:1].[CH2+:2][CH2:3]")
    bond_change = get_bond_change_nonexist_atoms(reactant, product)
    ref_bond_change = set()
    ref_bond_change.add((3, 2, 0.0))
    assert bond_change == ref_bond_change

    reactant = Chem.MolFromSmiles("[CH3:1][CH2+:2].[CH2:3][CH3:4]")
    product = Chem.MolFromSmiles("[CH3:1].[CH2+:2]")
    bond_change = get_bond_change_nonexist_atoms(reactant, product)
    ref_bond_change = set()
    ref_bond_change.add((2, 3, 1.0))
    assert bond_change == ref_bond_change

    reactant = Chem.MolFromSmiles("[CH3:1][CH2+:2].[CH2:3][CH2:4][CH3:5]")
    product = Chem.MolFromSmiles("[CH3:1][CH2+:2].[CH2:3]")
    bond_change = get_bond_change_nonexist_atoms(reactant, product)
    ref_bond_change = set()
    ref_bond_change.add((3, 2, 0.0))
    ref_bond_change.add((3, 4, 1.0))
    assert bond_change == ref_bond_change


def test_add_nonexist_atoms_and_bonds_to_product():

    reactant = Chem.MolFromSmiles("[CH3:1][CH2+:2].[CH2:3][CH2:4][CH3:5]")
    product = Chem.MolFromSmiles("[CH3:1][CH2+:2].[CH2:3]")
    bond_change = get_bond_change_nonexist_atoms(reactant, product)
    mol = add_nonexist_atoms_and_bonds_to_product(reactant, product, bond_change)
    mol_smi = Chem.MolToSmiles(mol)

    ref_smi = "[CH3:1][CH2+:2].[CH2:3].[CH3:4][CH3:5]"
    assert set(mol_smi.split(".")) == set(ref_smi.split("."))


def test_canonicalize_smiles_reaction_by_adding_nonexist_atoms_and_bonds():

    reaction = (
        "[CH3:1][CH2+:2].[CH2:3][CH2:4][CH3:5].[K+]>[Na+]>[CH3:1][CH2+:2].[CH2:3]"
    )
    out = canonicalize_smiles_reaction_by_adding_nonexist_atoms_and_bonds(reaction)
    canonical_rxn_smi, error = out
    reactant, reagent, product = canonical_rxn_smi.split(">")

    ref_reactant = "[CH3:1][CH2+:2].[CH2:3][CH2:4][CH3:5]"
    ref_product = "[CH3:1][CH2+:2].[CH2:3].[CH3:4][CH3:5]"
    ref_reagent = "[K+].[Na+]"

    assert error is None
    assert set(reactant.split(".")) == set(ref_reactant.split("."))
    assert set(product.split(".")) == set(ref_product.split("."))
    assert set(reagent.split(".")) == set(ref_reagent.split("."))
