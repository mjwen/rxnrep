import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from sklearn.utils import class_weight

from rxnrep.data.datamodule import BaseDataModule
from rxnrep.data.featurizer import (
    AtomFeaturizerMinimum2,
    BondFeaturizerMinimum,
    GlobalFeaturizer,
)
from rxnrep.data.io import read_mrnet_reaction_dataset
from rxnrep.data.uspto import BaseLabelledDataset

logger = logging.getLogger(__name__)


class ElectrolyteDataset(BaseLabelledDataset):
    """
    Electrolyte regression dataset.
    """

    def read_file(self, filename):
        return read_electrolyte_dataset(filename, self.nprocs)

    def generate_labels(self, normalize: bool = True):
        """
        Labels for all reactions.

        Args:
            normalize: whether to normalize `reaction_energy` and `activation_energy`
                labels.
        """

        # energies label
        reaction_energy = [
            rxn.get_property("reaction_energy") for rxn in self.reactions
        ]
        if not (set(reaction_energy) == {None}):
            reaction_energy = torch.as_tensor(reaction_energy, dtype=torch.float32)

            if normalize:
                reaction_energy = self.scale_label(
                    reaction_energy, name="reaction_energy"
                )

            # (each e is a scalar, but here we make it a 1D tensor of 1 element to use the
            # collate_fn, where all energies in a batch is cat to a 1D tensor)
            for i, rxn_e in enumerate(reaction_energy):
                self.labels[i]["reaction_energy"] = torch.as_tensor(
                    [rxn_e], dtype=torch.float32
                )

        activation_energy = [
            rxn.get_property("activation_energy") for rxn in self.reactions
        ]
        if not (set(activation_energy) == {None}):
            activation_energy = torch.as_tensor(activation_energy, dtype=torch.float32)

            if normalize:
                activation_energy = self.scale_label(
                    activation_energy, name="activation_energy"
                )

            # (each e is a scalar, but here we make it a 1D tensor of 1 element to use the
            # collate_fn, where all energies in a batch is cat to a 1D tensor)
            for i, act_e in enumerate(activation_energy):
                self.labels[i]["activation_energy"] = torch.as_tensor(
                    [act_e], dtype=torch.float32
                )


class ElectrolyteClassificationDataset(BaseLabelledDataset):
    def read_file(self, filename):
        return read_electrolyte_dataset(filename, self.nprocs)

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


class ElectrolyteClassificationDataModule(BaseDataModule):
    """
    Uspto dataset.
    """

    def __init__(
        self,
        trainset_filename: Union[str, Path],
        valset_filename: Union[str, Path],
        testset_filename: Union[str, Path],
        *,
        state_dict_filename: Union[str, Path] = "dataset_state_dict.yaml",
        restore_state_dict_filename: Optional[Union[str, Path]] = None,
        batch_size: int = 100,
        num_workers: int = 0,
        pin_memory: bool = True,
        num_processes: int = 1,
        build_reaction_graph: bool = True,
        num_reaction_classes: int = 2,
    ):
        super().__init__(
            trainset_filename,
            valset_filename,
            testset_filename,
            state_dict_filename=state_dict_filename,
            restore_state_dict_filename=restore_state_dict_filename,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            num_processes=num_processes,
            build_reaction_graph=build_reaction_graph,
        )

        self.num_reaction_classes = num_reaction_classes

    def setup(self, stage: Optional[str] = None):

        init_state_dict = self.get_init_state_dict()

        atom_featurizer = AtomFeaturizerMinimum2()
        bond_featurizer = BondFeaturizerMinimum()
        global_featurizer = GlobalFeaturizer(allowable_charge=[-1, 0, 1])

        self.data_train = ElectrolyteClassificationDataset(
            filename=self.trainset_filename,
            atom_featurizer=atom_featurizer,
            bond_featurizer=bond_featurizer,
            global_featurizer=global_featurizer,
            build_reaction_graph=self.build_reaction_graph,
            init_state_dict=init_state_dict,
            num_processes=self.num_processes,
            transform_features=True,
        )

        state_dict = self.data_train.state_dict()

        self.data_val = ElectrolyteClassificationDataset(
            filename=self.valset_filename,
            atom_featurizer=atom_featurizer,
            bond_featurizer=bond_featurizer,
            global_featurizer=global_featurizer,
            build_reaction_graph=self.build_reaction_graph,
            init_state_dict=state_dict,
            num_processes=self.num_processes,
            transform_features=True,
        )

        self.data_test = ElectrolyteClassificationDataset(
            filename=self.testset_filename,
            atom_featurizer=atom_featurizer,
            bond_featurizer=bond_featurizer,
            global_featurizer=global_featurizer,
            build_reaction_graph=self.build_reaction_graph,
            init_state_dict=state_dict,
            num_processes=self.num_processes,
            transform_features=True,
        )

        # save dataset state dict
        self.data_train.save_state_dict_file(self.state_dict_filename)

    def get_to_model_info(self) -> Dict[str, Any]:
        class_weight = self.data_train.get_class_weight(
            num_reaction_classes=self.num_reaction_classes, class_weight_as_1=True
        )

        d = {
            "feature_size": self.data_train.feature_size,
            "num_reaction_classes": self.num_reaction_classes,
            "reaction_class_weight": class_weight["reaction_type"],
        }

        return d


def read_electrolyte_dataset(filename, nprocs):
    logger.info("Start reading dataset file...")

    succeed_reactions, failed = read_mrnet_reaction_dataset(filename, nprocs)

    counter = Counter(failed)
    logger.info(
        f"Finish reading reactions. Number succeed {counter[False]}, "
        f"number failed {counter[True]}."
    )

    return succeed_reactions, failed
