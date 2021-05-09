import logging
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import torch
from sklearn.utils import class_weight

from rxnrep.data.datamodule import BaseDataModule
from rxnrep.data.dataset import (
    BaseContrastiveDataset,
    BaseLabelledDataset,
    ClassicalFeatureDataset,
)
from rxnrep.data.featurizer import AtomFeaturizer, BondFeaturizer, GlobalFeaturizer
from rxnrep.data.io import read_smiles_tsv_dataset

logger = logging.getLogger(__name__)


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


class GreenClassificationDataModule(BaseDataModule):
    """
    Green data module for classification.

    Args:
        num_reaction_classes: number of reaction class of the dataset. `None` means the
            dataset has no reaction type label.
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
        num_reaction_classes: Optional[int] = None,
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

        atom_featurizer_kwargs = {
            "atom_total_degree_one_hot": {"allowable_set": list(range(5))},
            "atom_total_valence_one_hot": {"allowable_set": list(range(5))},
        }

        atom_featurizer = AtomFeaturizer(featurizer_kwargs=atom_featurizer_kwargs)
        bond_featurizer = BondFeaturizer()
        global_featurizer = GlobalFeaturizer()

        self.data_train = GreenClassificationDataset(
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

        self.data_val = GreenClassificationDataset(
            filename=self.valset_filename,
            atom_featurizer=atom_featurizer,
            bond_featurizer=bond_featurizer,
            global_featurizer=global_featurizer,
            build_reaction_graph=self.build_reaction_graph,
            init_state_dict=state_dict,
            num_processes=self.num_processes,
            transform_features=True,
        )

        self.data_test = GreenClassificationDataset(
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

        logger.info(
            f"Trainset size: {len(self.data_train)}, valset size: {len(self.data_val)}: "
            f"testset size: {len(self.data_test)}."
        )

    def get_to_model_info(self) -> Dict[str, Any]:
        d = {"feature_size": self.data_train.feature_size}

        if self.num_reaction_classes:
            d["num_reaction_classes"] = self.num_reaction_classes

            class_weight = self.data_train.get_class_weight(
                num_reaction_classes=self.num_reaction_classes, class_weight_as_1=True
            )
            d["reaction_class_weight"] = class_weight["reaction_type"]

        return d


class GreenContrastiveDataModule(BaseDataModule):
    """
    Green datamodule for contrastive learning.

    Args:
        transform1: graph augmentation instance, see `transforms.py`
        transform2: graph augmentation instance, see `transforms.py`
    """

    def __init__(
        self,
        trainset_filename: Union[str, Path],
        valset_filename: Union[str, Path],
        testset_filename: Union[str, Path],
        *,
        transform1: Callable,
        transform2: Callable,
        state_dict_filename: Union[str, Path] = "dataset_state_dict.yaml",
        restore_state_dict_filename: Optional[Union[str, Path]] = None,
        batch_size: int = 100,
        num_workers: int = 0,
        pin_memory: bool = True,
        num_processes: int = 1,
        build_reaction_graph: bool = True,
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
        self.transform1 = transform1
        self.transform2 = transform2

    def setup(self, stage: Optional[str] = None):

        init_state_dict = self.get_init_state_dict()

        atom_featurizer_kwargs = {
            "atom_total_degree_one_hot": {"allowable_set": list(range(5))},
            "atom_total_valence_one_hot": {"allowable_set": list(range(5))},
        }

        atom_featurizer = AtomFeaturizer(featurizer_kwargs=atom_featurizer_kwargs)
        bond_featurizer = BondFeaturizer()
        global_featurizer = GlobalFeaturizer()

        self.data_train = GreenContrastiveDataset(
            filename=self.trainset_filename,
            atom_featurizer=atom_featurizer,
            bond_featurizer=bond_featurizer,
            global_featurizer=global_featurizer,
            build_reaction_graph=self.build_reaction_graph,
            init_state_dict=init_state_dict,
            num_processes=self.num_processes,
            transform_features=True,
            transform1=self.transform1,
            transform2=self.transform2,
        )

        state_dict = self.data_train.state_dict()

        self.data_val = GreenContrastiveDataset(
            filename=self.valset_filename,
            atom_featurizer=atom_featurizer,
            bond_featurizer=bond_featurizer,
            global_featurizer=global_featurizer,
            build_reaction_graph=self.build_reaction_graph,
            init_state_dict=state_dict,
            num_processes=self.num_processes,
            transform_features=True,
            transform1=self.transform1,
            transform2=self.transform2,
        )

        self.data_test = GreenContrastiveDataset(
            filename=self.testset_filename,
            atom_featurizer=atom_featurizer,
            bond_featurizer=bond_featurizer,
            global_featurizer=global_featurizer,
            build_reaction_graph=self.build_reaction_graph,
            init_state_dict=state_dict,
            num_processes=self.num_processes,
            transform_features=True,
            transform1=self.transform1,
            transform2=self.transform2,
        )

        # save dataset state dict
        self.data_train.save_state_dict_file(self.state_dict_filename)

        logger.info(
            f"Trainset size: {len(self.data_train)}, valset size: {len(self.data_val)}: "
            f"testset size: {len(self.data_test)}."
        )

    def get_to_model_info(self) -> Dict[str, Any]:
        d = {"feature_size": self.data_train.feature_size}

        return d
