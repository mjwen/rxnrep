import multiprocessing
import logging
import itertools
from pathlib import Path
from typing import Callable, Optional, Union, Dict


from pymatgen.reaction_network.reaction import Reaction as PMG_Reaction
from pymatgen.entries.mol_entry import MoleculeEntry
from monty.serialization import loadfn

from rxnrep.core.molecule import Molecule, MoleculeError
from rxnrep.core.reaction import Reaction
from rxnrep.core.rdmol import create_rdkit_mol_from_mol_graph
from rxnrep.data.uspto import USPTODataset

logger = logging.getLogger(__name__)


class ElectrolyteDataset(USPTODataset):
    """
    Electrolyte dataset for unsupervised reaction representation.

    Args:
        filename: tsv file of smiles reactions and labels
        atom_featurizer: function to create atom features
        bond_featurizer: function to create bond features
        global_featurizer: function to create global features
        transform_features: whether to standardize the atom, bond, and global features.
            If `True`, each feature column will first subtract the mean and then divide
            by the standard deviation.
        init_state_dict: initial state dict (or a yaml file of the state dict) containing
            the state of the dataset used for training: including all the atom types in
            the molecules, mean and stdev of the features (if transform_features is
            `True`). If `None`, these properties are computed from the current dataset.
        num_processes: number of processes used to load and process the dataset.
        return_index: whether to return the index of the sample in the dataset
    """

    def __init__(
        self,
        filename: Union[str, Path],
        atom_featurizer: Callable,
        bond_featurizer: Callable,
        global_featurizer: Callable,
        transform_features: bool = True,
        init_state_dict: Optional[Union[Dict, Path]] = None,
        num_processes: int = 1,
        return_index: bool = True,
    ):

        # read input files
        reactions, failed = self.read_file(filename, num_processes)

        super(USPTODataset, self).__init__(
            reactions,
            atom_featurizer,
            bond_featurizer,
            global_featurizer,
            init_state_dict,
            num_processes,
            return_index,
        )

        # set failed and labels
        self._failed = failed

        # convert reactions to dgl graphs
        self.reaction_graphs = self.build_graph_and_featurize()

        if transform_features:
            self.scale_features()

        self.labels = {}
        self.metadata = {}

    @staticmethod
    def read_file(filename, nprocs):

        # read file
        logger.info("Start reading dataset file...")

        pmg_reactions = loadfn(filename)

        logger.info("Finish reading dataset file...")

        # convert to reactions and labels
        logger.info("Start converting to reactions...")

        if nprocs == 1:
            reactions = [
                process_one_reaction_from_input_file(rxn) for rxn in pmg_reactions
            ]
        else:
            with multiprocessing.Pool(nprocs) as p:
                reactions = p.map(process_one_reaction_from_input_file, pmg_reactions)

        failed = []
        succeed_reactions = []
        for rxn in reactions:
            if rxn is None:
                failed.append(True)
            else:
                succeed_reactions.append(rxn)
                failed.append(False)

        logger.info("Finish converting to reactions...")

        return succeed_reactions, failed


def process_one_reaction_from_input_file(
    pmg_reaction: PMG_Reaction,
) -> Union[Reaction, None]:

    try:
        reaction = pymatgen_reaction_to_reaction(pmg_reaction)
    except MoleculeError:
        return None

    return reaction


def pymatgen_reaction_to_reaction(pmg_reaction: PMG_Reaction) -> Reaction:
    """
    Convert a pymatgen reaction to a rxnrep reaction.

    Args:
        pmg_reaction: pymatgen reaction

    Returns:
        a rxnrep reaction
    """
    # check map numbers are the same set in all the reactants and products
    reactants_map_numbers = [mp.values() for mp in pmg_reaction.reactants_atom_mapping]
    reactants_map_numbers = set(itertools.chain.from_iterable(reactants_map_numbers))
    products_map_numbers = [mp.values() for mp in pmg_reaction.products_atom_mapping]
    products_map_numbers = set(itertools.chain.from_iterable(products_map_numbers))
    if reactants_map_numbers != products_map_numbers:
        raise ValueError(
            "Expect atom map numbers to be the same set in the reactants and products; "
            f"got {reactants_map_numbers} for the reactants and {products_map_numbers} "
            "for the products."
        )

    # convert atom map number to 1 based if there is value smaller than 1
    min_val = min(reactants_map_numbers)
    if min_val < 1:
        converter = 1 - min_val
        reactants_atom_mapping = [
            {k: v + converter for k, v in mp.items()}
            for mp in pmg_reaction.reactants_atom_mapping
        ]
        products_atom_mapping = [
            {k: v + converter for k, v in mp.items()}
            for mp in pmg_reaction.products_atom_mapping
        ]
    else:
        reactants_atom_mapping = pmg_reaction.reactants_atom_mapping
        products_atom_mapping = pmg_reaction.products_atom_mapping

    reactants = [
        pymatgen_mol_entry_to_molecule(entry, mapping)
        for entry, mapping in zip(pmg_reaction.reactants, reactants_atom_mapping)
    ]

    products = [
        pymatgen_mol_entry_to_molecule(entry, mapping)
        for entry, mapping in zip(pmg_reaction.products, products_atom_mapping)
    ]

    reactant_ids = "+".join([str(i) for i in pmg_reaction.reactant_ids])
    product_ids = "+".join([str(i) for i in pmg_reaction.product_ids])
    reaction_id = reactant_ids + " -> " + product_ids

    reaction = Reaction(reactants, products, id=reaction_id)

    return reaction


def pymatgen_mol_entry_to_molecule(
    mol_entry: MoleculeEntry, atom_mapping: Dict[int, int]
) -> Molecule:
    """
    Convert a pymatgen molecule entry to a rxnrep molecule, and set the atom mapping.

    Args:
        mol_entry: molecule entry
        atom_mapping: {atom_index: map_number} atom map number for atoms in the molecule

    Returns:
        rxnrep molecule with atom mapping
    """

    sorted_atom_index = sorted(atom_mapping.keys())

    # check has atom map number for each atom
    if sorted_atom_index != list(range(mol_entry.num_atoms)):
        raise ValueError(
            f"Incorrect `atom_mapping`. Molecule has {mol_entry.num_atoms} atoms, "
            f"but provided mapping number are for atoms {sorted_atom_index}."
        )

    rdkit_mol, _ = create_rdkit_mol_from_mol_graph(mol_entry.mol_graph)

    # set rdkit mol atom map number
    for i in sorted_atom_index:
        map_number = atom_mapping[i]
        rdkit_mol.GetAtomWithIdx(i).SetAtomMapNum(map_number)

    # NOTE, The total charge of the created rdkit molecule may be different from the
    # charge of the pymatgen Molecule. Currently, there is not a good method to
    # handle metals. We set the charge of Molecule to what is in pymatgen Molecule
    # explicitly.
    mol = Molecule(rdkit_mol, id=mol_entry.entry_id)
    mol.charge = mol_entry.molecule.charge

    return mol
