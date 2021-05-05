import logging
from collections import Counter
from pathlib import Path
from typing import Dict

import torch
from sklearn.utils import class_weight

from rxnrep.data.dataset import (
    BaseContrastiveDataset,
    BaseLabelledDataset,
    ClassicalFeatureDataset,
)
from rxnrep.data.io import read_smiles_tsv_dataset

logger = logging.getLogger(__name__)


class GreenDataset(BaseLabelledDataset):
    """
    Green reaction activation energy dataset.
    """

    def read_file(self, filename: Path):
        return read_green_dataset(filename, self.nprocs)

    def generate_labels(self, normalize: bool = True):
        """
        Labels for all reactions.

        Args:
            normalize: whether to normalize the reaction energy and activation energy
                labels
        """

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


class GreenClassificationDataset(BaseLabelledDataset):
    def read_file(self, filename: Path):
        return read_green_dataset(filename, self.nprocs)

    def generate_labels(self):
        """
        `reaction_type` label.
        """
        for i, rxn in enumerate(self.reactions):
            rxn_class = rxn.get_property("reaction_type")
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

        weight = {"reaction_type": w}

        return weight


class GreenContrastiveDataset(BaseContrastiveDataset):
    def read_file(self, filename: Path):
        return read_green_dataset(filename, self.nprocs)


class GreenClassicalFeaturesDataset(ClassicalFeatureDataset):
    def read_file(self, filename: Path):
        return read_green_dataset(filename, self.nprocs)

    def generate_labels(self):
        """
        Labels for all reactions.

        Add `reaction_type`.
        """

        labels = []
        for i, rxn in enumerate(self.reactions):
            rxn_class = rxn.get_property("reaction_type")
            labels.append(
                {"reaction_type": torch.as_tensor(int(rxn_class), dtype=torch.int64)}
            )
        return labels


def read_green_dataset(filename: Path, nprocs):
    logger.info("Start reading dataset ...")

    succeed_reactions, failed = read_smiles_tsv_dataset(
        filename, remove_H=False, nprocs=nprocs
    )

    counter = Counter(failed)
    logger.info(
        f"Finish reading dataset. Number succeed {counter[False]}, "
        f"number failed {counter[True]}."
    )

    return succeed_reactions, failed
