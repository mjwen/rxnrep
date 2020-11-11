import multiprocessing
import logging
import itertools
from pathlib import Path
from collections import Counter
from typing import Callable, Optional, Union, Dict, Tuple

import torch
import numpy as np

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

        logger.info(
            f"Finish converting to reactions. Number succeed {len(succeed_reactions)}, "
            f"number failed {Counter(failed)[True]}."
        )

        return succeed_reactions, failed


class ElectrolyteDatasetTwoBondType(ElectrolyteDataset):
    """
    Similar to ElectrolyteDataset, the difference is that here we only have two bond
    types: unchanged or changed.

    This is supposed to be used for A->B and A->B+C reactions where there is only bond
    breakage, not bond creation.

    """

    def get_atom_in_reaction_center_and_bond_type_class_weight(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create class weight to be used in cross entropy function for node classification
        problems:

        1. whether atom is in reaction center:
            weight = mean(num_atoms_not_in_center / num_atoms_in_center)
            where mean is taken over all reactions.

            Here, if an atom is in center, it has positive class 1, we adjust its
            weight accordingly. See `self._create_label_atom_in_reaction_center()` for
            more.

        2. bond is unchanged bond, changed bond (lost or added bond):
            weight = mean(num_unchanged_bonds / num_changed_bonds)
            where mean is taken over all reactions.

            Here, unchanged and changed have class labels 0 and 1
            respectively. See `self._create_label_bond_type()` for more.

        Returns:
            weight_atom_in_reaction_center: a scaler tensor giving the weight for the
                positive class.
            weight_bond_type: a scaler tensor giving the weight for the positive class.
        """

        w_in_center = []
        w_changed_bond = []
        for rxn in self.reactions:
            unchanged, lost, added = rxn.get_unchanged_lost_and_added_bonds(
                zero_based=True
            )

            # bond weight
            n_unchanged = len(unchanged)
            n_changed = len(lost + added)
            if n_changed == 0:
                w_changed_bond.append(1.0)
            else:
                w_changed_bond.append(n_unchanged / n_changed)

            # atom weight
            num_atoms = sum([m.num_atoms for m in rxn.reactants])
            changed_atoms = set(
                [i for i in itertools.chain.from_iterable(lost + added)]
            )
            num_changed = len(changed_atoms)
            if num_changed == 0:
                w_in_center.append(1.0)
            else:
                w_in_center.append((num_atoms - num_changed) / num_changed)

        weight_atom_in_reaction_center = torch.as_tensor(
            np.mean(w_in_center), dtype=torch.float32
        )

        weight_bond_type = torch.as_tensor(np.mean(w_changed_bond), dtype=torch.float32)

        return weight_atom_in_reaction_center, weight_bond_type

    @staticmethod
    def _create_label_bond_type(reaction) -> torch.Tensor:
        """
        Label for bond type classification:
        0: unchanged bond, 1: changed bond (lost or added bond)

        Args:
            reaction: the reaction

        Returns:
            1D tensor of the class for each bond. The order is the same as the bond
            nodes in the reaction graph.
       """
        result = reaction.get_num_unchanged_lost_and_added_bonds()
        num_unchanged, num_lost, num_added = result

        # Note, reaction graph bond nodes are ordered in the sequence of unchanged bonds,
        # lost bonds, and added bonds in `create_reaction_graph()`
        bond_type = [0] * num_unchanged + [1] * (num_lost + num_added)
        bond_type = torch.as_tensor(bond_type, dtype=torch.float32)

        return bond_type


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
