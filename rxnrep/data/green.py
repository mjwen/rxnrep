import logging
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import torch
from sklearn.utils import class_weight

from rxnrep.data.dataset import (
    BaseContrastiveDataset,
    BaseDatasetWithLabels,
    ClassicalFeatureDataset,
)
from rxnrep.data.io import read_smiles_tsv_dataset

logger = logging.getLogger(__name__)


class GreenDataset(BaseDatasetWithLabels):
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
        build_reaction_graph=True,
        init_state_dict: Optional[Union[Dict, Path]] = None,
        transform_features: bool = True,
        return_index: bool = True,
        num_processes: int = 1,
        #
        # args to control labels
        #
        allow_label_scaler_none: bool = False,
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
            build_reaction_graph=build_reaction_graph,
            init_state_dict=init_state_dict,
            transform_features=transform_features,
            return_index=return_index,
            num_processes=num_processes,
            allow_label_scaler_none=allow_label_scaler_none,
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


class GreenContrastiveDataset(BaseContrastiveDataset):
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


class GreenClassicalFeaturesDataset(ClassicalFeatureDataset):
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

    def generate_labels(self):
        """
        Labels for all reactions.

        Add `reaction_type`.
        """

        labels = []
        for i, rxn in enumerate(self.reactions):
            rxn_class = rxn.get_property("class label")
            labels.append(
                {"reaction_type": torch.as_tensor(int(rxn_class), dtype=torch.int64)}
            )
        return labels


class GreenClassificationDataset(BaseDatasetWithLabels):
    def __init__(
        self,
        filename: Union[str, Path],
        atom_featurizer: Callable,
        bond_featurizer: Callable,
        global_featurizer: Callable,
        *,
        build_reaction_graph=True,
        init_state_dict: Optional[Union[Dict, Path]] = None,
        transform_features: bool = True,
        return_index: bool = True,
        num_processes: int = 1,
    ):

        super().__init__(
            filename,
            atom_featurizer,
            bond_featurizer,
            global_featurizer,
            build_reaction_graph=build_reaction_graph,
            init_state_dict=init_state_dict,
            transform_features=transform_features,
            return_index=return_index,
            num_processes=num_processes,
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

    def generate_labels(self):
        """
        Labels for all reactions.

        Add `reaction_type`.
        """
        super().generate_labels()

        for i, rxn in enumerate(self.reactions):
            rxn_class = rxn.get_property("class label")
            self.labels[i]["reaction_type"] = torch.as_tensor(
                [int(rxn_class)], dtype=torch.int64
            )

    def get_class_weight(
        self, num_reaction_classes: int = None, class_weight_as_1: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Create class weight to be used in cross entropy losses.

        Args:
            num_reaction_classes: number of reaction classes in the dataset. The class
            labels should be 0, 1, 2, ... num_reaction_classes-1.
            class_weight_as_1: If `True`, the weight for all classes is set to 1.0;
                otherwise, it is inversely proportional to the number of data points in
                the dataset
        """
        weight = super().get_class_weight(only_break_bond=False)

        if class_weight_as_1:
            w = torch.ones(num_reaction_classes)
        else:
            rxn_classes = [rxn.get_property("class label") for rxn in self.reactions]

            # class weight for reaction classes
            w = class_weight.compute_class_weight(
                "balanced",
                classes=list(range(num_reaction_classes)),
                y=rxn_classes,
            )
            w = torch.as_tensor(w, dtype=torch.float32)

        weight["reaction_type"] = w

        return weight
