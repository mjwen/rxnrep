import logging
from collections import Counter
from pathlib import Path
from typing import Dict

import torch
from sklearn.utils import class_weight

from rxnrep.data.dataset import BaseDatasetWithLabels
from rxnrep.data.io import read_smiles_tsv_dataset

logger = logging.getLogger(__name__)


class USPTODataset(BaseDatasetWithLabels):
    """
    USPTO dataset.
    """

    def read_file(self, filename: Path):
        logger.info("Start reading dataset ...")

        succeed_reactions, failed = read_smiles_tsv_dataset(
            filename, remove_H=True, nprocs=self.nprocs
        )

        counter = Counter(failed)
        logger.info(
            f"Finish reading dataset. Number succeed {counter[False]}, "
            f"number failed {counter[True]}."
        )

        return succeed_reactions, failed


class SchneiderDataset(USPTODataset):
    """
    Schneider 50k USPTO dataset with class labels for reactions.

    The difference between this and the USPTO dataset is that there is class label in
    this dataset and no class label in USPTO. This is added as the `reaction_class`
    in the `labels`.
    """

    def generate_labels(self):
        """
        Labels for all reactions.

        Add `reaction_class`.
        """
        super().generate_labels()

        for i, rxn in enumerate(self.reactions):
            rxn_class = rxn.get_property("label")
            self.labels[i]["reaction_class"] = torch.as_tensor(
                [int(rxn_class)], dtype=torch.int64
            )

    def get_class_weight(
        self, num_reaction_classes: int = 50, class_weight_as_1: bool = False
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
        # class weight for atom hop and bond hop
        if self.max_hop_distance:
            weight = super().get_class_weight()
        else:
            weight = {}

        if class_weight_as_1:
            w = torch.ones(num_reaction_classes)
        else:
            rxn_classes = [rxn.get_property("label") for rxn in self.reactions]

            # class weight for reaction classes
            w = class_weight.compute_class_weight(
                "balanced",
                classes=list(range(num_reaction_classes)),
                y=rxn_classes,
            )
            w = torch.as_tensor(w, dtype=torch.float32)

        weight["reaction_class"] = w

        return weight
