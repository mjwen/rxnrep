import logging
import multiprocessing
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import pandas as pd
import torch

from rxnrep.core.molecule import MoleculeError
from rxnrep.core.reaction import Reaction, ReactionError, smiles_to_reaction
from rxnrep.data.uspto import USPTODataset
from rxnrep.utils import to_path

logger = logging.getLogger(__name__)


class GreenDataset(USPTODataset):
    """
    Green reaction activation energy dataset.

    Args:
        have_activation_energy_ratio: a randomly selecgted portion of this amount of
            reactions will be marked to have activation energies, and the others not.
    """

    def __init__(
        self,
        filename: Union[str, Path],
        atom_featurizer: Callable,
        bond_featurizer: Callable,
        global_featurizer: Callable,
        *,
        init_state_dict: Optional[Union[Dict, Path]] = None,
        transform_features: bool = True,
        return_index: bool = True,
        num_processes: int = 1,
        # args to control labels
        max_hop_distance: int = 2,
        atom_type_masker_ratio: Union[float, None] = None,
        atom_type_masker_use_masker_value: bool = True,
        # ratio of activation energy label to use
        have_activation_energy_ratio=1.0,
    ):
        pass

        super().__init__(
            filename,
            atom_featurizer,
            bond_featurizer,
            global_featurizer,
            init_state_dict=init_state_dict,
            transform_features=transform_features,
            return_index=return_index,
            num_processes=num_processes,
            max_hop_distance=max_hop_distance,
            atom_type_masker_ratio=atom_type_masker_ratio,
            atom_type_masker_use_masker_value=atom_type_masker_use_masker_value,
        )

        self.have_activation_energy_ratio = have_activation_energy_ratio
        self.have_activation_energy = self.generate_have_activation_energy(
            have_activation_energy_ratio
        )

    @staticmethod
    def read_file(filename: Path, nprocs: int):

        # read file
        logger.info("Start reading dataset file...")

        filename = to_path(filename)
        df = pd.read_csv(filename, sep="\t")
        smiles_reactions = df["reaction"].tolist()

        logger.info("Finish reading dataset file...")

        # convert to reactions
        logger.info("Start converting to reactions...")

        ids = [f"{smi}_index-{i}" for i, smi in enumerate(smiles_reactions)]
        if nprocs == 1:
            reactions = [
                process_one_reaction_from_input_file(smi, i)
                for smi, i in zip(smiles_reactions, ids)
            ]
        else:
            args = zip(smiles_reactions, ids)
            with multiprocessing.Pool(nprocs) as p:
                reactions = p.starmap(process_one_reaction_from_input_file, args)

        # column names besides `reaction`
        column_names = df.columns.values.tolist()
        column_names.remove("reaction")

        succeed_reactions = []
        failed = []

        for i, rxn in enumerate(reactions):
            if rxn is None:
                failed.append(True)
            else:
                # keep other info (e.g. label) in input file as reaction property
                for name in column_names:
                    rxn.set_property(name, df[name][i])

                succeed_reactions.append(rxn)
                failed.append(False)

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
            `atom_hop_dist`, `bond_hop_dist`, `reaction_energy`, `activation_energy`.

        Args:
            normalize: whether to normalize the reaction energy and activation energy
                labels
        """

        # `atom_hop_dist` and `bond_hop_dist` labels
        labels = super().generate_labels()

        # `reaction_energy` and `activation_energy` label

        reaction_energy = torch.as_tensor(
            [rxn.get_property("reaction enthalpy") for rxn in self.reactions],
            dtype=torch.float32,
        )
        activation_energy = torch.as_tensor(
            [rxn.get_property("activation energy") for rxn in self.reactions],
            dtype=torch.float32,
        )
        if normalize:
            reaction_energy = self.scale_label(reaction_energy, name="reaction_energy")
            activation_energy = self.scale_label(
                activation_energy, name="activation_energy"
            )

        # (each energy is a scalar, but here we make it a 1D tensor of 1 element to use
        # the collate_fn, where all energies in a batch is cat to a 1D tensor)
        for re, ae, rxn_label in zip(reaction_energy, activation_energy, labels):
            rxn_label["reaction_energy"] = torch.as_tensor([re], dtype=torch.float32)
            rxn_label["activation_energy"] = torch.as_tensor([ae], dtype=torch.float32)

        return labels

    def generate_have_activation_energy(self, ratio: float) -> torch.Tensor:
        """
        Mark a portion of reactions to have activation energy and the others not.

        Args:
            ratio: the ratio of of reactions to have activation energy

        Returns:
            1D tensor of size N, where N is the number of data points (reactions).
                Each element is a bool tensor indicating whether activation energy
                exists for the reaction or not.

        """
        n = len(self.reactions)

        activation_energy_exist = torch.zeros(n, dtype=torch.bool)

        # randomly selected indices to make as exists
        selected = torch.randperm(n)[: int(n * ratio)]

        activation_energy_exist[selected] = True

        return activation_energy_exist

    def __getitem__(self, item):
        out = super().__getitem__(item)
        if self.return_index:
            item, reactants_g, products_g, reaction_g, meta, label = out
        else:
            reactants_g, products_g, reaction_g, meta, label = out

        meta["have_activation_energy"] = self.have_activation_energy[item]

        if self.return_index:
            return item, reactants_g, products_g, reaction_g, meta, label
        else:
            return reactants_g, products_g, reaction_g, meta, label

    @staticmethod
    def collate_fn(samples):
        (
            batched_indices,
            batched_molecule_graphs,
            batched_reaction_graphs,
            batched_labels,
            batched_metadata,
        ) = super(GreenDataset, GreenDataset).collate_fn(samples)

        batched_metadata["have_activation_energy"] = torch.as_tensor(
            batched_metadata["have_activation_energy"]
        )

        return (
            batched_indices,
            batched_molecule_graphs,
            batched_reaction_graphs,
            batched_labels,
            batched_metadata,
        )

    def get_property(self, name: str):
        """
        Get property for all data points.
        """
        if name in ["reaction_energy", "activation_energy"]:
            return torch.cat([lb[name] for lb in self.labels])
        elif name == "have_activation_energy":
            return torch.as_tensor(self.have_activation_energy)
        else:
            raise ValueError(f"Unsupported property name {name}")


def process_one_reaction_from_input_file(
    smiles_reaction: str, id: str
) -> Union[Reaction, None]:
    """
    Helper function to create reactions using multiprocessing.

    Note, not remove H from smiles.
    """

    try:
        reaction = smiles_to_reaction(
            smiles_reaction,
            id=id,
            ignore_reagents=True,
            remove_H=False,
            sanity_check=False,
        )
    except (MoleculeError, ReactionError):
        return None

    return reaction
