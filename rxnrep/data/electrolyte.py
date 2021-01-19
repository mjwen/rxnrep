import itertools
import logging
import multiprocessing
from collections import Counter
from typing import Dict, List, Union

import numpy as np
import torch
from monty.serialization import loadfn
from mrnet.core.mol_entry import MoleculeEntry
from mrnet.core.reactions import Reaction as PMG_Reaction
from sklearn.utils import class_weight

from rxnrep.core.molecule import Molecule, MoleculeError
from rxnrep.core.rdmol import create_rdkit_mol_from_mol_graph
from rxnrep.core.reaction import Reaction, ReactionError
from rxnrep.data.uspto import USPTODataset

logger = logging.getLogger(__name__)


class ElectrolyteDataset(USPTODataset):
    """
    Electrolyte dataset for unsupervised reaction representation.
    """

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
                process_one_reaction_from_input_file(rxn, i)
                for i, rxn in enumerate(pmg_reactions)
            ]
        else:
            ids = list(range(len(pmg_reactions)))
            args = zip(pmg_reactions, ids)
            with multiprocessing.Pool(nprocs) as p:
                reactions = p.starmap(process_one_reaction_from_input_file, args)

        failed = []
        succeed_reactions = []
        for rxn in reactions:
            if rxn is None:
                failed.append(True)
            else:
                failed.append(False)
                succeed_reactions.append(rxn)

        counter = Counter(failed)
        logger.info(
            f"Finish converting to reactions. Number succeed {counter[False]}, "
            f"number failed {counter[True]}."
        )

        return succeed_reactions, failed

    def generate_labels(self, normalize: bool = True) -> List[Dict[str, torch.Tensor]]:
        """
        Labels for all reactions.

        Each dict is the labels for one reaction, with keys:
            `atom_hop_dist`, `bond_hop_dist`, and `reaction_energy`.

        Args:
            normalize: whether to normalize `reaction_energy` labels
        """

        # `atom_hop_dist` and `bond_hop_dist` labels
        labels = super().generate_labels()

        # `reaction_energy` label
        energies = torch.as_tensor(
            [rxn.get_property("free_energy") for rxn in self.reactions],
            dtype=torch.float32,
        )
        if normalize:
            energies = self.scale_label(energies, name="reaction_energy")

        # (each e is a scalar, but here we make it a 1D tensor of 1 element to use the
        # collate_fn, where all energies in a batch is cat to a 1D tensor)
        for e, rxn_label in zip(energies, labels):
            rxn_label["reaction_energy"] = torch.as_tensor([e], dtype=torch.float32)

        return labels


class ElectrolyteDatasetNoAddedBond(ElectrolyteDataset):
    """
    Similar to ElectrolyteDataset, the difference is that here all the reactions are
    one bond breaking reactions (A->B and A->B+C), and there is no bond creation.

    As a result, allowed number of classes for atom hop distance and bond hop distances
    are both different from ElectrolyteDataset.
    """

    def get_class_weight(self) -> Dict[str, torch.Tensor]:
        """
        Create class weight to be used in cross entropy losses.

        This is for labels generated in `generate_labels()`.
        For each type of, it is computed as the mean over all reactions.
        """

        # atom hop class weight

        # Unique labels should be `list(range(atom_hop_num_classes))`,  where
        # `atom_hop_num_classes` should be `max_hop_distance + 1`.
        # The labels are: atoms in lost bond (class 0) and atoms in unchanged bond (
        # class 1 to max_hop_distance).
        all_atom_hop_labels = np.concatenate(
            [lb["atom_hop_dist"] for lb in self.labels]
        )

        unique_labels = sorted(set(all_atom_hop_labels))
        if unique_labels != list(range(self.max_hop_distance + 1)):
            raise RuntimeError(
                f"Unable to compute atom class weight; some classes do not have valid "
                f"labels. num_classes: {self.max_hop_distance + 1} unique labels: "
                f"{unique_labels}"
            )

        atom_hop_weight = class_weight.compute_class_weight(
            "balanced",
            classes=unique_labels,
            y=all_atom_hop_labels,
        )

        # bond hop class weight
        # Unique labels should be `list(range(bond_hop_num_classes))`, where
        # `bond_hop_num_classes = max_hop_distance + 1`.
        # The labels are: lost bond (class 0), unchanged (class 1 to max_hop_distance).
        all_bond_hop_labels = np.concatenate(
            [lb["bond_hop_dist"] for lb in self.labels]
        )

        unique_labels = sorted(set(all_bond_hop_labels))
        if unique_labels != list(range(self.max_hop_distance + 1)):
            raise RuntimeError(
                f"Unable to compute bond class weight; some classes do not have valid "
                f"labels. num_classes: {self.max_hop_distance + 1} unique labels: "
                f"{unique_labels}"
            )

        bond_hop_weight = class_weight.compute_class_weight(
            "balanced",
            classes=unique_labels,
            y=all_bond_hop_labels,
        )

        weight = {
            "atom_hop_dist": torch.as_tensor(atom_hop_weight, dtype=torch.float32),
            "bond_hop_dist": torch.as_tensor(bond_hop_weight, dtype=torch.float32),
        }

        return weight


def process_one_reaction_from_input_file(
    pmg_reaction: PMG_Reaction, index: int
) -> Union[Reaction, None]:

    try:
        reaction = pymatgen_reaction_to_reaction(pmg_reaction, index)
    except (MoleculeError, ReactionError):
        return None

    return reaction


def pymatgen_reaction_to_reaction(pmg_reaction: PMG_Reaction, index: int) -> Reaction:
    """
    Convert a pymatgen reaction to a rxnrep reaction.

    Args:
        pmg_reaction: pymatgen reaction
        index: index of the reaction in the whole dataset

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
    reaction_id = f"{reactant_ids}->{product_ids}_index-{index}"

    # additional property
    free_energy = sum([m.get_property("free_energy") for m in products]) - sum(
        [m.get_property("free_energy") for m in reactants]
    )

    reaction = Reaction(
        reactants,
        products,
        id=reaction_id,
        sanity_check=False,
        properties={"free_energy": free_energy},
    )

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
    mol = Molecule(
        rdkit_mol,
        id=mol_entry.entry_id,
        properties={"free_energy": mol_entry.get_free_energy()},
    )
    mol.charge = mol_entry.molecule.charge

    return mol
