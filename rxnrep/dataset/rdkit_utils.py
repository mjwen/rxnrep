from rdkit.Chem import Mol
from typing import List, Tuple, Dict


def get_mol_atom_mapping(m: Mol) -> List[int]:
    """
    Get atom mapping of an rdkit molecule.

    Returns atom mapping for each atom. `None` if the atom is not mapped.

    """
    ind_map = []
    for atom in m.GetAtoms():
        map_num = atom.GetAtomMapNum()
        if map_num != 0:
            ind_map.append(map_num - 1)
        else:
            ind_map.append(None)

    return ind_map


def get_reaction_atom_mapping(
    reactants: List[Mol], products: List[Mol]
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
