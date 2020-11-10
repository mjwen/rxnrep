import copy
import multiprocessing
import numpy as np
from rdkit import Chem
from rdkit.Chem import (
    KekulizeException,
    AtomKekulizeException,
    AtomValenceException,
    MolSanitizeException,
)
from typing import List, Tuple, Dict, Union, Set, Any


def get_atom_property(m: Chem.Mol, include_H: bool = True) -> List[Dict[str, Any]]:
    """
    Get the property (e.g. valence, number H) of all atoms in a molecule.

    Args:
        m: rdkit molecule
        include_H: whether to get the properties for H atoms

    Returns:
        [{property_name, property_value}], each dict holds the property of an atom.
    """
    prop = []
    for i, atom in enumerate(m.GetAtoms()):
        if not include_H and atom.GetSymbol() == "H":
            continue
        atom_prop = {
            "atom_index": i,
            "atom_map_number": atom.GetAtomMapNum(),
            "specie": atom.GetSymbol(),
            "num_implicit_H": atom.GetNumImplicitHs(),
            "num_explicit_H": atom.GetNumExplicitHs(),
            "total_num_H(include_graph_H)": atom.GetTotalNumHs(includeNeighbors=True),
            "total_num_H(exclude_graph_H)": atom.GetTotalNumHs(includeNeighbors=False),
            "implicit_valence": atom.GetImplicitValence(),
            "explict_valence": atom.GetExplicitValence(),
            "total_valence": atom.GetTotalValence(),
            "num_radicals": atom.GetNumRadicalElectrons(),
            "formal_charge": atom.GetFormalCharge(),
            "is_aromatic": atom.GetIsAromatic(),
        }
        prop.append(atom_prop)

    return prop


def get_atom_property_as_dict(
    mol: Chem.Mol, include_H: bool = True
) -> Dict[int, Dict[str, Any]]:
    """
    Get the property (e.g. valence, number H) of all atoms in a molecule, and return a
    dict with atom map number as the key to index the atom.

    Args:
        mol: rdkit molecule to get properties

    Returns:
        {atom_map_number: {property_name, property_value}}, each inner dict holds the
        property of an atom, with atom map number of the atom as the key.
    """
    props_of_all_atoms = get_atom_property(mol, include_H)

    props_dict = {}
    for atom_props in props_of_all_atoms:
        atom_map_number = atom_props["atom_map_number"]
        if atom_map_number is None:
            raise AtomMapNumberError(
                "Cannot create atom property dict for molecule some atoms of which do "
                "not have atom map number."
            )
        props_dict[atom_map_number] = atom_props

    return props_dict


def check_molecule_atom_mapped(m: Chem.Mol) -> bool:
    """
    Check if all atoms in a molecule are mapped.

    Args:
        m: rdkit mol
    """
    for atom in m.GetAtoms():
        if not atom.HasProp("molAtomMapNumber"):
            return False

    return True


def check_reaction_atom_mapped(reaction: str) -> Tuple[bool, bool]:
    """
    Check both reactants and products of a reaction are atom mapped.

    Args:
        reaction: smiles representation of a reaction.

    Returns:
        reactants_mapped: whether reactants is mapped, `None` if molecule cannot be
            created.
        products_mapped: whether products is mapped, `None` if molecule cannot be created.
    """
    reactants, _, products = reaction.strip().split(">")
    rct = Chem.MolFromSmiles(reactants)
    prdt = Chem.MolFromSmiles(products)

    if rct is None or prdt is None:
        return None, None

    else:
        reactants_mapped = check_molecule_atom_mapped(rct)
        products_mapped = check_molecule_atom_mapped(prdt)

        return reactants_mapped, products_mapped


def check_all_reactions_atom_mapped(
    reactions: List[str], nprocs=1, print_result=False
) -> List[Tuple[bool, bool]]:
    """
    Check that reactants and products in all reactions are atom mapped.

    Args:
        reactions: list of smiles representation of a reaction
        nprocs: number of processes to use
        print_result: whether to print out results

    Returns:
        mapped: mapped[i] is a two-tuple indicating whether the reactants and products
        are mapped.
    """
    if nprocs == 1:
        mapped = [check_reaction_atom_mapped(rxn) for rxn in reactions]
    else:
        with multiprocessing.Pool(nprocs) as p:
            mapped = p.map(check_reaction_atom_mapped, reactions)

    if print_result:
        if np.all(mapped):
            print("Reactants and products in all reactions are mapped.")
        else:
            n_failed = 0
            n_rct = 0
            n_prdt = 0
            for i, mp in enumerate(mapped):

                if mp[0] is None or mp[1] is None:
                    n_failed += 1
                elif not mp[0] and not mp[1]:
                    n_rct += 1
                    n_prdt += 1
                    print(f"{i} both reactants and products are not mapped")
                elif not mp[0]:
                    n_rct += 1
                    print(f"{i} reactants are not mapped")
                elif not mp[1]:
                    n_prdt += 1
                    print(f"{i} reactants are not mapped")
            print(
                f"Total number of reactions: {len(mapped)}; reactants not mapped: "
                f"{n_rct}; products not mapped {n_prdt}; "
                f"molecules cannot be converted: {n_failed}."
            )

        print("Done!")

    return mapped


def check_bonds_mapped(m: Chem.Mol) -> Tuple[bool, bool]:
    """
    Check whether all bonds in a molecule are mapped.

    Returns:
        has_bond_both_atoms_not_mapped
        has_bond_one_atom_not_mapped
    """
    has_bond_both_atoms_not_mapped = False
    has_bond_one_atom_not_mapped = False
    for bond in m.GetBonds():
        atom1_mapped = bond.GetBeginAtom().HasProp("molAtomMapNumber")
        atom2_mapped = bond.GetEndAtom().HasProp("molAtomMapNumber")
        if has_bond_both_atoms_not_mapped and has_bond_one_atom_not_mapped:
            break
        # both not mapped
        if not atom1_mapped and not atom2_mapped:
            has_bond_both_atoms_not_mapped = True
        # one mapped, the other not
        elif atom1_mapped != atom2_mapped:
            has_bond_one_atom_not_mapped = True

    return has_bond_both_atoms_not_mapped, has_bond_one_atom_not_mapped


def check_reaction_bonds_mapped(reaction: str) -> Tuple[bool, bool]:
    """
    Check the atom mapping for bonds in the reactants of a reaction.

    Args:
        reaction: smiles representation of a reaction.

    Returns:
        has_bond_both_atoms_not_mapped
        has_bond_one_atom_not_mapped
    """
    reactants, _, products = reaction.strip().split(">")
    rct = Chem.MolFromSmiles(reactants)
    prdt = Chem.MolFromSmiles(products)

    if rct is None or prdt is None:
        return None, None

    else:
        has_bonds_both_not_mapped, has_bonds_one_not_mapped = check_bonds_mapped(rct)

        return has_bonds_both_not_mapped, has_bonds_one_not_mapped


def check_all_reactions_bonds_mapped(
    reactions: List[str], nprocs=1, print_result=False
) -> List[Tuple[bool, bool]]:
    """
    Check that reactants and products in all reactions are bond mapped.
    """
    if nprocs == 1:
        mapped = [check_reaction_bonds_mapped(rxn) for rxn in reactions]
    else:
        with multiprocessing.Pool(nprocs) as p:
            mapped = p.map(check_reaction_bonds_mapped, reactions)

    if print_result:
        if np.all(mapped):
            print("All bonds are mapped.")
        else:
            n_failed = 0
            n_both = 0
            n_one = 0
            for i, mp in enumerate(mapped):

                if mp[0] is None or mp[1] is None:
                    n_failed += 1
                elif mp[0] and mp[1]:
                    n_both += 1
                    print(f"{i} has bonds both atoms not mapped")
                elif mp[0] != mp[1]:
                    n_one += 1
                    print(f"{i} has bonds one atom not mapped")
            print(
                f"Total number of reactions: {len(mapped)}; reactions having bonds "
                f"both atoms not mapped: {n_both}; having bonds one atom not mapped "
                f" {n_one}; molecules cannot be converted: {n_failed}."
            )

        print("Done!")

    return mapped


def canonicalize_smiles_reaction(
    reaction: str,
) -> Tuple[Union[str, None], Union[None, str]]:
    """
    Canonicalize a smiles reaction to make reactants and products have the same
    composition.

    This ensures the reactants and products have the same composition, achieved in the
    below steps:

    1. remove reactant molecules from reactants if none of their atoms are present in
       the products
    2. adjust atom mapping between reactants and products and add atom mapping number
       for reactant atoms without a mapping number (although there is no corresponding
       atom in the products)
    3. create new products by editing the reactants: removing bonds in the reactants
       but not in the products and adding bonds not in the reactants but in the products

    Args:
        reaction: smiles representation of a reaction

    Returns:
        reaction: canonicalized smiles reaction, `None` if canonicalize failed
        error: error message, `None` if canonicalize succeed
    """

    # Step 1, adjust reagents
    try:
        rxn_smi = adjust_reagents(reaction)
    except (MoleculeCreationError, AtomMapNumberError) as e:
        return None, str(e).rstrip()

    # Step 2, adjust atom mapping
    try:
        reactants_smi, reagents_smi, products_smi = rxn_smi.strip().split(">")
        reactants = Chem.MolFromSmiles(reactants_smi)
        products = Chem.MolFromSmiles(products_smi)
        reactants = set_no_graph_H(reactants)
        products = set_no_graph_H(products)
        reactants, products = adjust_atom_map_number(reactants, products)
    except AtomMapNumberError as e:
        return None, str(e).rstrip()

    # Step 3, create new products by editing bonds of the reactants. Some atom properties
    # (formal charge, and number of radicals) are copied from the products, though

    try:
        bond_changes, has_lost, has_added, _ = get_reaction_bond_change(
            reactants, products
        )

        # check 3.1, skip reactions only has bond type changes, but no bond lost or added
        # (e.g. add H to benzene)
        if not (has_lost or has_added):
            return None, "reactions with only bond type changes"

        product_atom_properties = get_atom_property_as_dict(products)
        new_products = edit_molecule(reactants, bond_changes, product_atom_properties)

        # check 3.2, remove reactions that break a bond with H, and produce a product of H2.
        # In cases where the orientation of H is specified with `\` or `/` (e.g. as in
        # "[H]/[CH]=N/[H]"), the H will be always in the molecule graph. Then, breaking
        # such a bond will produce an H2, but one H in H2 will not be mapped.
        # So, here we check all products atoms are mapped.
        if not check_molecule_atom_mapped(new_products):
            return None, "products after mol editing have atoms not mapped"

    except (KekulizeException, AtomKekulizeException, AtomValenceException) as e:
        return None, str(e).rstrip()

    # write canonicalized reaction to smiles
    try:
        reactants_smi = Chem.MolToSmiles(set_all_H_to_explicit(reactants))
        products_smi = Chem.MolToSmiles(set_all_H_to_explicit(new_products))
    except MolSanitizeException as e:
        return None, str(e).rstrip()
    canoical_reaction = ">".join([reactants_smi, reagents_smi, products_smi])

    return canoical_reaction, None


def canonicalize_smiles_reaction_by_adding_nonexist_atoms_and_bonds(
    reaction: str,
) -> Tuple[Union[str, None], Union[None, str]]:
    """
    Canonicalize a smiles reaction to make reactants and products have the same
    composition.

    This is an alternative of `canonicalize_smiles_reaction()`.
    In `canonicalize_smiles_reaction()`. bond connectivity of the reactant is edited to
    simulate the reaction to obtain the product. A problem with this method is that it
    edits many unnecessary bonds (e.g. bonds that change bond type). This function uses
    a differnet approach:

    1. remove reactant molecules from reactants if none of their atoms are present in
       the products
    2. adjust atom mapping between reactants and products and add atom mapping number
       for reactant atoms without a mapping number (although there is no corresponding
       atom in the products)
    3. add atoms in the reactant but not in the product to the product, and add bonds
       associated with these atoms to the product. Note, we only add bonds that both
       the atoms are not in product. Bond with one atom in the product one atom not in
       the prouduct is regared as a breaking bond and thus not added to the product.

    Args:
        reaction: smiles representation of a reaction

    Returns:
        reaction: canonicalized smiles reaction, `None` if canonicalize failed
        error: error message, `None` if canonicalize succeed
    """

    # Step 1, adjust reagents
    try:
        rxn_smi = adjust_reagents(reaction)
    except (MoleculeCreationError, AtomMapNumberError) as e:
        return None, "Step 1 " + str(e).rstrip()

    # Step 2, adjust atom mapping
    try:
        reactants_smi, reagents_smi, products_smi = rxn_smi.strip().split(">")
        reactants = Chem.MolFromSmiles(reactants_smi)
        products = Chem.MolFromSmiles(products_smi)
        reactants = set_no_graph_H(reactants)
        products = set_no_graph_H(products)
        reactants, products = adjust_atom_map_number(reactants, products)
    except AtomMapNumberError as e:
        return None, "Step 2 " + str(e).rstrip()

    # Step 3, create new products by editing bonds of the reactants. Some atom properties
    # (formal charge, and number of radicals) are copied from the products, though

    # check 3.1, skip reactions only has bond type changes, but no bond lost or added
    # (e.g. add H to benzene)
    if not check_connectivity_change(reactants, products):
        return None, "Step 3.1 reactions with only bond type changes"

    try:
        bond_changes = get_bond_change_nonexist_atoms(reactants, products)
        new_products = add_nonexist_atoms_and_bonds_to_product(
            reactants, products, bond_changes
        )
    except (KekulizeException, AtomKekulizeException, AtomValenceException) as e:
        return None, "Step 3 " + str(e).rstrip()

    # check 3.2, remove reactions that break a bond with H, and produce a product of
    # H2. In cases where the orientation of H is specified with `\` or `/` (e.g.
    # "[H]/[CH]=N/[H]"), the H will be always in the molecule graph. Then, breaking
    # such a bond will produce an H2, but one H in H2 will not be mapped.
    # So, here we check all products atoms are mapped.
    if not check_molecule_atom_mapped(new_products):
        return None, "Step 3.2 products after mol editing have atoms not mapped"

    # Step 4, write canonicalized reaction to smiles
    try:
        reactants_smi = Chem.MolToSmiles(reactants)
        products_smi = Chem.MolToSmiles(new_products)
    except MolSanitizeException as e:
        return None, "Step 4 " + str(e).rstrip()
    canoical_reaction = ">".join([reactants_smi, reagents_smi, products_smi])

    return canoical_reaction, None


def get_mol_atom_mapping(m: Chem.Mol) -> List[int]:
    """
    Get atom mapping for an rdkit molecule.

    Args:
        m: rdkit molecule

    Returns:
         atom mapping for each atom. `None` if the atom is not mapped.
    """
    mapping = []
    for atom in m.GetAtoms():
        if atom.HasProp("molAtomMapNumber"):
            mapping.append(atom.GetAtomMapNum())
        else:
            mapping.append(None)
    return mapping


def adjust_reagents(reaction: str) -> str:
    """
    Move reagents in the reactants or products to the reagents collection.

    For a smiles reaction of the type `aaa>bbb>ccc`, aaa is the reactants, bbb is the
    reagents, and ccc is the products. It could happen that some molecule in aaa (ccc)
    does not have a single atom in ccc (aaa). Such molecules should actually be reagents.
    This function moves such molecules from aaa (ccc) to bbb.

    Args:
        reaction: smiles representation of an atom mapped reaction

    Returns:
         smiles reaction with the place of reagents adjusted
    """

    reactants_smi, reagents_smi, products_smi = reaction.strip().split(">")

    reactants = [Chem.MolFromSmiles(s) for s in reactants_smi.split(".")]
    products = [Chem.MolFromSmiles(s) for s in products_smi.split(".")]
    if None in reactants or None in products:
        raise MoleculeCreationError(f"Cannot create molecules from: {reaction}")

    # get atom mapping
    mapping_rcts = [set(get_mol_atom_mapping(m)) for m in reactants]
    mapping_prdts = [set(get_mol_atom_mapping(m)) for m in products]
    mapping_rcts_all = set()
    mapping_prdts_all = set()
    for x in mapping_rcts:
        mapping_rcts_all.update(x)
    for x in mapping_prdts:
        mapping_prdts_all.update(x)
    if None in mapping_prdts_all:
        raise AtomMapNumberError("Products has atom without map number.")

    new_reactants = []
    new_reagents = []
    new_products = []
    # move reactant to reagent if none of its atoms is in the product
    for i, mapping in enumerate(mapping_rcts):
        if len(mapping & mapping_prdts_all) == 0:
            new_reagents.append(reactants[i])
        else:
            new_reactants.append(reactants[i])

    # move product to reagent if none of its atoms is in the reactant
    for i, mapping in enumerate(mapping_prdts):
        if len(mapping & mapping_rcts_all) == 0:
            new_reagents.append(products[i])
        else:
            new_products.append(products[i])

    # remove atom mapping in new reagents
    for m in new_reagents:
        for a in m.GetAtoms():
            a.ClearProp("molAtomMapNumber")

    reactants_smi = ".".join([Chem.MolToSmiles(m) for m in new_reactants])
    products_smi = ".".join([Chem.MolToSmiles(m) for m in new_products])
    if reagents_smi == "":
        reagents_smi = ".".join([Chem.MolToSmiles(m) for m in new_reagents])
    else:
        reagents_smi = ".".join(
            [reagents_smi] + [Chem.MolToSmiles(m) for m in new_reagents]
        )

    reaction = ">".join([reactants_smi, reagents_smi, products_smi])

    return reaction


def adjust_atom_map_number(
    reactant: Chem.Mol, product: Chem.Mol
) -> Tuple[Chem.Mol, Chem.Mol]:
    """
    Adjust atom map number between the reactant and product.

    The below steps are performed:

    1. Check the map numbers are unique for the reactants (products), i.e. each map
       number only occur once such that no map number is associated with two atoms.
    2. Check whether all product atoms are mapped to reactant atoms. If not, throw an
       error. It is not required that all reactant atoms should be mapped.
    3. Renumber the existing atom map numbers to let it start from 1 and be consecutive.
    4. Add atom map numbers to atoms in the reactant if they do not have one.

    The input reactant and product are not be modified.

    Args:
        reactant: rdkit molecule
        product: rdkit molecule

    Returns:
        reactant and product with atom map number adjusted
    """

    reactant = copy.deepcopy(reactant)
    product = copy.deepcopy(product)

    rct_mapping = get_mol_atom_mapping(reactant)
    prdt_mapping = get_mol_atom_mapping(product)

    # Step 1, check map number uniqueness
    if None in prdt_mapping:
        raise AtomMapNumberError("Products has atom without map number.")
    if len(prdt_mapping) != len(set(prdt_mapping)):
        raise AtomMapNumberError("Products has atoms with the same map number.")
    rct_mapping_no_None = [mp for mp in rct_mapping if mp is not None]
    if len(rct_mapping_no_None) != len(set(rct_mapping_no_None)):
        raise AtomMapNumberError("Reactants has atoms with the same map number.")

    # Step 2, check all product atoms are mapped
    if not set(prdt_mapping).issubset(set(rct_mapping)):
        raise AtomMapNumberError("Products has atom not mapped to reactant.")

    # Step 3, Renumber existing atom map

    # clear reactant atom map number
    for a in reactant.GetAtoms():
        a.ClearProp("molAtomMapNumber")

    # set atom map number
    for i, mp in enumerate(prdt_mapping):
        rct_atom_idx = rct_mapping.index(mp)
        prdt_atom_idx = i
        reactant.GetAtomWithIdx(rct_atom_idx).SetAtomMapNum(i + 1)
        product.GetAtomWithIdx(prdt_atom_idx).SetAtomMapNum(i + 1)

    # Step 4, add atom map number for reactant atoms does not have one
    i = len(prdt_mapping)
    for a in reactant.GetAtoms():
        if not a.HasProp("molAtomMapNumber"):
            a.SetAtomMapNum(i + 1)
            i += 1

    return reactant, product


def get_reaction_bond_change(
    reactant: Chem.Mol, product: Chem.Mol, use_mapped_atom_index=False
) -> Tuple[Set[Tuple[int, int, float]], bool, bool, bool]:
    """
    Get the changes of the bonds to make the products from the reactants.

    Bond changes include:

    1. lost bond: bonds in reactants but not in products (note bonds whose both atoms
        are not in the product are not considered)
    2  added bond: bonds not in reactants but in products
    3. altered bond: bonds in both reactants and products, both their bond type changes

    Args:
        reactant: rdkit molecule
        product: rdkit molecule
        use_mapped_atom_index: this determines what to use for the atom index in the
            returned bond changes. If `False`, using the atom index in the underlying
            rdkit molecule; if `True`, using the mapped atom index.

    Returns:
        bond_change: each element is a three-tuple (atom_1, atom_2, change_type) denoting
            the change of a bond. `atom_1` and `atom_2` are indices of the two atoms
            forming the bond. The atom indices could either be the non-mapped or the
            mapped indices, depending on `use_mapped_atom_index`.
            `change_type` can take 0, 1, 2, 3, and 1.5, meaning losing a bond, forming
            a single, double, triple, and aromatic bond, respectively.
        has_lost: whether there is bond lost in the reaction
        has_added: whether there is bond added in the reaction
        has_altered: whether there is bond that changes bond type in the reaction
    """

    reactant_map_numbers = get_mol_atom_mapping(reactant)
    product_map_numbers = get_mol_atom_mapping(product)

    # bonds in reactant (only consider bonds at least one atom is in the product)
    bonds_rct = {}
    for bond in reactant.GetBonds():
        bond_atoms = [bond.GetBeginAtom(), bond.GetEndAtom()]
        if (bond_atoms[0].GetAtomMapNum() in product_map_numbers) or (
            bond_atoms[1].GetAtomMapNum() in product_map_numbers
        ):
            num_pair = tuple(sorted([a.GetAtomMapNum() for a in bond_atoms]))
            bonds_rct[num_pair] = bond.GetBondTypeAsDouble()

    # bonds in product (only consider bonds at least one atom is in the reactant)
    bonds_prdt = {}
    for bond in product.GetBonds():
        bond_atoms = [bond.GetBeginAtom(), bond.GetEndAtom()]
        if (bond_atoms[0].GetAtomMapNum() in reactant_map_numbers) or (
            bond_atoms[1].GetAtomMapNum() in reactant_map_numbers
        ):
            num_pair = tuple(sorted([a.GetAtomMapNum() for a in bond_atoms]))
            bonds_prdt[num_pair] = bond.GetBondTypeAsDouble()

    bond_changes = set()

    has_lost = False
    has_added = False
    has_altered = False
    for bond in bonds_rct:
        if bond not in bonds_prdt:
            # lost bond
            bond_changes.add((bond[0], bond[1], 0.0))
            has_lost = True
        else:
            if bonds_rct[bond] != bonds_prdt[bond]:
                # changed bond
                bond_changes.add((bond[0], bond[1], bonds_prdt[bond]))
                has_altered = True

    for bond in bonds_prdt:
        if bond not in bonds_rct:
            # new bond
            bond_changes.add((bond[0], bond[1], bonds_prdt[bond]))
            has_added = True

    # convert mapped atom index to the underlying rdkit atom index (non-mapped)
    # of the reactant
    if not use_mapped_atom_index:
        atom_mp = get_mol_atom_mapping(reactant)
        converter = {v: i for i, v in enumerate(atom_mp) if v is not None}
        bond_changes_new_atom_index = []
        for atom1, atom2, change in bond_changes:
            idx1, idx2 = sorted([converter[atom1], converter[atom2]])
            bond_changes_new_atom_index.append((idx1, idx2, change))

        bond_changes = set(bond_changes_new_atom_index)

    return bond_changes, has_lost, has_added, has_altered


def edit_molecule(
    mol: Chem.Mol,
    edits: Set[Tuple[int, int, float]],
    atom_props: Dict[int, Dict[str, Any]],
) -> Chem.Mol:
    """
    Edit a molecule to generate a new one by applying the bond changes.

    Args:
        mol: rdkit molecule to edit
        edits: each element is a three-tuple (atom_1, atom_2, change_type) denoting
            the change of a bond. `atom_1` and `atom_2` are indices of the two atoms
            forming the bond, and `change_type` can take 0, 1, 2, 3, and 1.5, meaning
            losing a bond, forming a single, double, triple, and aromatic bond,
            respectively.
        atom_props: {atom_map_number: {property_name, property_value}}. Here the
            `num_radicals` and `formal_charge` in the property dict is set for each atom.

    Returns:
        new_mol: a new molecule after applying the bond edits to the input molecule
    """

    bond_change_to_type = {
        1.0: Chem.rdchem.BondType.SINGLE,
        2.0: Chem.rdchem.BondType.DOUBLE,
        3.0: Chem.rdchem.BondType.TRIPLE,
        1.5: Chem.rdchem.BondType.AROMATIC,
    }

    # Let all explicit H be implicit. This increases the number of implicit H and
    # allows the adjustment of the number of implicit H to satisfy valence rule
    mol = set_all_H_to_implicit(mol)

    # set the formal charge and number of radicals
    for a in mol.GetAtoms():
        map_number = a.GetAtomMapNum()
        # only set the properties for atoms whose properties are given
        if map_number in atom_props:
            a.SetFormalCharge(atom_props[map_number]["formal_charge"])
            a.SetNumRadicalElectrons(atom_props[map_number]["num_radicals"])

    # editing bonds
    rw_mol = Chem.RWMol(mol)
    for atom1, atom2, change_type in edits:
        bond = rw_mol.GetBondBetweenAtoms(atom1, atom2)
        if bond is not None:
            rw_mol.RemoveBond(atom1, atom2)
        if change_type > 0:
            rw_mol.AddBond(atom1, atom2, bond_change_to_type[change_type])
    new_mol = rw_mol.GetMol()

    # Sanitize after editing
    # adjust aromatic for atoms on longer in ring
    for atom in new_mol.GetAtoms():
        if not atom.IsInRing():
            atom.SetIsAromatic(False)
    # other sanitize (e.g. adjust num implicit H)
    Chem.SanitizeMol(new_mol)

    # After editing, we set all hydrogen to explicit
    new_mol = set_all_H_to_explicit(new_mol)

    return new_mol


def get_bond_change_nonexist_atoms(
    reactant: Chem.Mol, product: Chem.Mol, use_mapped_atom_index=False
):
    """
    Get the bond change from reactant to products, only for bonds
    1. both the two atoms are missing in the products or
    2. one of the two atoms are missing in the products

    Args:
        reactant: rdkit molecule
        product: rdkit molecule
        use_mapped_atom_index: this determines what to use for the atom index in the
            returned bond changes. If `False`, using the atom index in the underlying
            rdkit molecule; if `True`, using the mapped atom index.

    Returns:
        bond_change: each element is a three-tuple (atom_1, atom_2, change_type) denoting
            the change of a bond. `atom_1` and `atom_2` are indices of the two atoms
            forming the bond. The atom indices could either be the non-mapped or the
            mapped indices, depending on `use_mapped_atom_index`.
            `change_type` can take 0, 1, 2, 3, and 1.5.
            When change_type = 1, 2, 3, or 1.5, it means both the two atoms forming the
            bonds are not in the product, and 1, 2, 3, and 1.5 are the bond type in the
            reactant.
            When change_type = 0, it means atom_1 is in the product but atom_2 is not.
            This also means this bond should not exist in the product.
    """

    product_map_numbers = get_mol_atom_mapping(product)

    bond_changes = set()
    for bond in reactant.GetBonds():
        bond_atoms = [bond.GetBeginAtom(), bond.GetEndAtom()]
        map_number = tuple(
            sorted([bond_atoms[0].GetAtomMapNum(), bond_atoms[1].GetAtomMapNum()])
        )

        # both atoms not in product
        if (map_number[0] not in product_map_numbers) and (
            map_number[1] not in product_map_numbers
        ):
            bond_changes.add((map_number[0], map_number[1], bond.GetBondTypeAsDouble()))

        # one atom not in product
        elif (
            map_number[0] not in product_map_numbers
            and map_number[1] in product_map_numbers
        ):
            bond_changes.add((map_number[0], map_number[1], 0.0))

        elif (
            map_number[0] in product_map_numbers
            and map_number[1] not in product_map_numbers
        ):
            bond_changes.add((map_number[1], map_number[0], 0.0))

    # convert mapped atom index to the underlying rdkit atom index (non-mapped)
    # of the reactant
    if not use_mapped_atom_index:
        atom_mp = get_mol_atom_mapping(reactant)
        converter = {v: i for i, v in enumerate(atom_mp) if v is not None}
        bond_changes_new_atom_index = []
        for atom1, atom2, change in bond_changes:
            bond_changes_new_atom_index.append(
                (converter[atom1], converter[atom2], change)
            )

        bond_changes = set(bond_changes_new_atom_index)

    return bond_changes


def check_connectivity_change(reactant: Chem.Mol, product: Chem.Mol) -> bool:
    """
    Determine whether there is bond connectivity change in the reaction.

    Args:
        reactant:
        product:

    Returns:
        `True` if graph connectivity changes from reactant to product; otherwise `False`
        i.e. only bond type changes.
    """
    reactant_bonds = set()
    for bond in reactant.GetBonds():
        bond_atoms = [bond.GetBeginAtom(), bond.GetEndAtom()]
        map_number = tuple(
            sorted([bond_atoms[0].GetAtomMapNum(), bond_atoms[1].GetAtomMapNum()])
        )
        reactant_bonds.add(map_number)

    product_bonds = set()
    for bond in product.GetBonds():
        bond_atoms = [bond.GetBeginAtom(), bond.GetEndAtom()]
        map_number = tuple(
            sorted([bond_atoms[0].GetAtomMapNum(), bond_atoms[1].GetAtomMapNum()])
        )
        product_bonds.add(map_number)

    return not reactant_bonds == product_bonds


def add_nonexist_atoms_and_bonds_to_product(
    reactant: Chem.Mol, product: Chem.Mol, bond_change: Set[Tuple[int, int, float]]
) -> Chem.Mol:
    """
    Add atoms that are in the reactant but not in the product (and the associated
    bonds) to the products.

    Args:
        reactant: rdkit molecule
        product: rdkit molecule
        bond_change: each element is a three-tuple (atom_1, atom_2, change_type) denoting
            the change of a bond. `atom_1` and `atom_2` are indices of the two atoms
            forming the bond.
            `change_type` can take 0, 1, 2, 3, and 1.5.
            When change_type = 1, 2, 3, or 1.5, it means both the two atoms forming the
            bonds are not in the product, and 1, 2, 3, and 1.5 are the bond type in the
            reactant.
            When change_type = 0, it means atom_1 is in the product but atom_2 is not.
            This also means this bond should not exist in the product.
    """

    # all atoms in the reactant are in the product; no need to edit the product
    if not bond_change:
        return product

    bond_change_to_type = {
        1.0: Chem.rdchem.BondType.SINGLE,
        2.0: Chem.rdchem.BondType.DOUBLE,
        3.0: Chem.rdchem.BondType.TRIPLE,
        1.5: Chem.rdchem.BondType.AROMATIC,
    }

    # atoms not the products
    non_exist_atoms = set()
    for atom1, atom2, change_type in bond_change:
        if change_type == 0.0:
            non_exist_atoms.add(atom1)
        else:
            non_exist_atoms.update([atom1, atom2])

    # add atoms not in the products to the products
    atom_properties = get_atom_property_as_dict(reactant)
    reactant_map_numbers = get_mol_atom_mapping(reactant)

    rw_mol = Chem.RWMol(product)
    for i in sorted(non_exist_atoms):
        specie = reactant.GetAtomWithIdx(i).GetSymbol()
        atom = Chem.Atom(specie)
        map_number = reactant_map_numbers[i]
        # set atom properties
        atom.SetAtomMapNum(map_number)
        atom.SetFormalCharge(atom_properties[map_number]["formal_charge"])
        atom.SetNumRadicalElectrons(atom_properties[map_number]["num_radicals"])
        rw_mol.AddAtom(atom)

    # get map dict between reactant atom index and product atom index
    product_map_numbers = get_mol_atom_mapping(rw_mol)
    reactant_2_product = {}
    for i, map_number in enumerate(reactant_map_numbers):
        reactant_2_product[i] = product_map_numbers.index(map_number)

    # add bonds (only for bonds whose both atoms are not in product)
    for atom1, atom2, change_type in bond_change:
        if change_type > 0:
            atom1 = reactant_2_product[atom1]
            atom2 = reactant_2_product[atom2]
            rw_mol.AddBond(atom1, atom2, bond_change_to_type[change_type])
    new_mol = rw_mol.GetMol()

    # sanitize mol
    for atom in new_mol.GetAtoms():
        if not atom.IsInRing():
            atom.SetIsAromatic(False)
    Chem.SanitizeMol(new_mol)

    return new_mol


def set_no_graph_H(m: Chem.Mol) -> Chem.Mol:
    """
    Set H in graph to implicit for explicit H.

    Args:
        m: rdkit molecule

    Returns:
        updated molecule with no graph H

    """
    m2 = Chem.RemoveHs(m, implicitOnly=False)
    Chem.SanitizeMol(m2)
    return m2


def set_all_H_to_implicit(m: Chem.Mol) -> Chem.Mol:
    """
    Set all the hydrogens on atoms to implicit.


    Args:
        m: rdkit molecule

    Returns rdkit molecule with all hydrogens implicit
    """
    m2 = Chem.RemoveHs(m, implicitOnly=False)
    for atom in m2.GetAtoms():
        atom.SetNoImplicit(False)
        atom.SetNumExplicitHs(0)
    Chem.SanitizeMol(m2)

    return m2


def set_all_H_to_explicit(m: Chem.Mol) -> Chem.Mol:
    """
    Set all the hydrogens on atoms to explicit.


    Args:
        m: rdkit molecule

    Returns rdkit molecule with all hydrogens explicit
    """

    # method 1
    # m2 = Chem.RemoveHs(m, implicitOnly=False)
    # for atom in m2.GetAtoms():
    #     num_H = atom.GetTotalNumHs()
    #     atom.SetNoImplicit(True)
    #     atom.SetNumExplicitHs(num_H)

    # method 2
    m2 = Chem.AddHs(m, explicitOnly=False)
    for atom in m2.GetAtoms():
        atom.SetNoImplicit(True)
    m2 = Chem.RemoveHs(m2, implicitOnly=False)
    Chem.SanitizeMol(m2)

    return m2


def get_reaction_atom_mapping(
    reactants: List[Chem.Mol], products: List[Chem.Mol]
) -> List[Dict[int, Tuple[int, int]]]:
    """
    Create atom mapping between reactants and products.

    Each dictionary is the mapping for a reactant, in the format:
     {atom_index: {product_index, product_atom_index}}.

    If a mapping cannot be found for an atom in a reactant molecule, it is set to (
    None, None).
    """
    reactants_mp = [get_mol_atom_mapping(m) for m in reactants]
    products_mp = [get_mol_atom_mapping(m) for m in products]

    mappings = []
    for rct_mp in reactants_mp:
        molecule = {}
        for atom_idx, atom_mp in enumerate(rct_mp):
            for prdt_idx, prdt_mp in enumerate(products_mp):
                if atom_mp in prdt_mp:
                    idx = (prdt_idx, prdt_mp.index(atom_mp))
                    break
            else:
                idx = (None, None)
            molecule[atom_idx] = idx

        mappings.append(molecule)

    return mappings


class MoleculeCreationError(Exception):
    def __init__(self, msg):
        super(MoleculeCreationError, self).__init__(msg)
        self.msg = msg


class AtomMapNumberError(Exception):
    def __init__(self, msg):
        super(AtomMapNumberError, self).__init__(msg)
        self.msg = msg
