import logging
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import torch

from rxnrep.data.io import read_smiles_tsv_dataset
from rxnrep.data.uspto import USPTODataset

logger = logging.getLogger(__name__)


class GreenDataset(USPTODataset):
    """
    Green reaction activation energy dataset.

    Args:
        have_activation_energy_ratio: a randomly selected portion of this amount of
            reactions will be marked to have activation energies, and the others not.
            If None, all will have activation energy.
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
        #
        # args to control labels
        #
        max_hop_distance: Optional[int] = None,
        atom_type_masker_ratio: Optional[float] = None,
        atom_type_masker_use_masker_value: Optional[bool] = None,
        have_activation_energy_ratio: Optional[float] = None,
    ):
        self.have_activation_energy_ratio = have_activation_energy_ratio

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

    def read_file(self, filename: Path):
        logger.info("Start reading dataset ...")

        succeed_reactions, failed = read_smiles_tsv_dataset(
            filename, remove_H=False, nprocs=self.nprocs
        )

        counter = Counter(failed)
        logger.info(
            f"Finish reading dataset. Number succeed {counter[False]}, "
            f"number failed {counter[True]}."
        )

        return succeed_reactions, failed

    def generate_labels(self, normalize: bool = True):
        """
        Labels for all reactions.

        Each dict is the labels for one reaction, with keys:
            `atom_hop_dist`, `bond_hop_dist`, `reaction_energy`, `activation_energy`.

        Args:
            normalize: whether to normalize the reaction energy and activation energy
                labels
        """
        super().generate_labels()

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
        for i, (re, ae) in enumerate(zip(reaction_energy, activation_energy)):
            self.labels[i].update(
                {
                    "reaction_energy": torch.as_tensor([re], dtype=torch.float32),
                    "activation_energy": torch.as_tensor([ae], dtype=torch.float32),
                }
            )

    def generate_metadata(self):
        """
        Added: `have_activation_energy`.
        """
        super().generate_metadata()

        if self.have_activation_energy_ratio is not None:
            act_energy_exists = generate_have_activation_energy(
                len(self.reactions), self.have_activation_energy_ratio
            )
            for i, x in enumerate(act_energy_exists):
                self.medadata[i]["have_activation_energy"] = x

    @staticmethod
    def collate_fn(samples):
        (
            batched_indices,
            batched_molecule_graphs,
            batched_reaction_graphs,
            batched_labels,
            batched_metadata,
        ) = super(GreenDataset, GreenDataset).collate_fn(samples)

        if "have_activation_energy" in batched_metadata:
            batched_metadata["have_activation_energy"] = torch.stack(
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
            return torch.stack([m[name] for m in self.medadata])
        else:
            raise ValueError(f"Unsupported property name {name}")


def generate_have_activation_energy(n: int, ratio: float) -> torch.Tensor:
    """
    Mark a portion of reactions to have activation energy and the others not.

    Args:
        n: total number of reactions
        ratio: the ratio of of reactions to have activation energy

    Returns:
        1D tensor of size N, where N is the number of data points (reactions).
            Each element is a bool tensor indicating whether activation energy
            exists for the reaction or not.

    """

    activation_energy_exist = torch.zeros(n, dtype=torch.bool)

    # randomly selected indices to make as exists
    selected = torch.randperm(n)[: int(n * ratio)]

    activation_energy_exist[selected] = True

    return activation_energy_exist
