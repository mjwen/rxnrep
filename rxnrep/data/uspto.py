import logging
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import torch
from sklearn.utils import class_weight
from torch.utils.data import DataLoader

from rxnrep.data.datamodule import BaseDataModule
from rxnrep.data.dataset import (
    BaseContrastiveDataset,
    BaseLabelledDataset,
    ClassicalFeatureDataset,
)
from rxnrep.data.featurizer import (
    AtomFeaturizer,
    BondFeaturizer,
    GlobalFeaturizer,
    MorganFeaturizer,
)
from rxnrep.data.io import read_smiles_tsv_dataset

logger = logging.getLogger(__name__)


def read_uspto_file(filename: Path, nprocs):
    logger.info("Start reading dataset ...")

    succeed_reactions, failed = read_smiles_tsv_dataset(
        filename, remove_H=True, nprocs=nprocs
    )

    counter = Counter(failed)
    logger.info(
        f"Finish reading dataset. Number succeed {counter[False]}, "
        f"number failed {counter[True]}."
    )

    return succeed_reactions, failed


class USPTODataset(BaseLabelledDataset):
    """
    USPTO dataset.

    Args:
        has_class_label: whether the dataset provides class label. The Schneider dataset
            has.
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
        has_class_label: bool = False,
    ):
        self.has_class_label = has_class_label

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
        return read_uspto_file(filename, self.nprocs)

    def generate_labels(self):
        """
        Create `reaction_type` label.
        """
        if self.has_class_label:
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

        if self.has_class_label:
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

            weight = {"reaction_type": w}
        else:
            weight = {}

        return weight


class USPTOContrastiveDataset(BaseContrastiveDataset):
    def read_file(self, filename: Path):
        return read_uspto_file(filename, self.nprocs)


class USPTOClassicalFeaturesDataset(ClassicalFeatureDataset):
    def read_file(self, filename: Path):
        return read_uspto_file(filename, self.nprocs)

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
            rxn_classes = [rxn.get_property("label") for rxn in self.reactions]

            # class weight for reaction classes
            w = class_weight.compute_class_weight(
                "balanced",
                classes=list(range(num_reaction_classes)),
                y=rxn_classes,
            )
            w = torch.as_tensor(w, dtype=torch.float32)

        weight = {"reaction_type": w}

        return weight


class UsptoDataModule(BaseDataModule):
    """
    Uspto dataset.

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
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            num_processes=num_processes,
            state_dict_filename=state_dict_filename,
            restore_state_dict_filename=restore_state_dict_filename,
            build_reaction_graph=build_reaction_graph,
        )

        self.num_reaction_classes = num_reaction_classes

    def setup(self, stage: Optional[str] = None):

        init_state_dict = self.get_init_state_dict()

        has_class_label = self.num_reaction_classes is not None

        atom_featurizer = AtomFeaturizer()
        bond_featurizer = BondFeaturizer()
        global_featurizer = GlobalFeaturizer()

        self.data_train = USPTODataset(
            filename=self.trainset_filename,
            atom_featurizer=atom_featurizer,
            bond_featurizer=bond_featurizer,
            global_featurizer=global_featurizer,
            build_reaction_graph=self.build_reaction_graph,
            init_state_dict=init_state_dict,
            num_processes=self.num_processes,
            transform_features=True,
            # label args
            has_class_label=has_class_label,
        )

        state_dict = self.data_train.state_dict()

        self.data_val = USPTODataset(
            filename=self.valset_filename,
            atom_featurizer=atom_featurizer,
            bond_featurizer=bond_featurizer,
            global_featurizer=global_featurizer,
            build_reaction_graph=self.build_reaction_graph,
            init_state_dict=state_dict,
            num_processes=self.num_processes,
            transform_features=True,
            # label args
            has_class_label=has_class_label,
        )

        self.data_test = USPTODataset(
            filename=self.testset_filename,
            atom_featurizer=atom_featurizer,
            bond_featurizer=bond_featurizer,
            global_featurizer=global_featurizer,
            build_reaction_graph=self.build_reaction_graph,
            init_state_dict=state_dict,
            num_processes=self.num_processes,
            transform_features=True,
            # label args
            has_class_label=has_class_label,
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


class UsptoContrastiveDataModule(BaseDataModule):
    """
    Uspto datamodule for contrastive learning.

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
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            num_processes=num_processes,
            state_dict_filename=state_dict_filename,
            restore_state_dict_filename=restore_state_dict_filename,
            build_reaction_graph=build_reaction_graph,
        )
        self.transform1 = transform1
        self.transform2 = transform2

    def setup(self, stage: Optional[str] = None):

        init_state_dict = self.get_init_state_dict()

        atom_featurizer = AtomFeaturizer()
        bond_featurizer = BondFeaturizer()
        global_featurizer = GlobalFeaturizer()

        self.data_train = USPTOContrastiveDataset(
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

        self.data_val = USPTOContrastiveDataset(
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

        self.data_test = USPTOContrastiveDataset(
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


class UsptoMorganDataModule(BaseDataModule):
    """
    Uspto datamodule using Morgan feats.

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
        batch_size: int = 100,
        num_workers: int = 0,
        pin_memory: bool = True,
        num_processes: int = 1,
        num_reaction_classes: Optional[int] = None,
        morgan_radius: int = 2,
        morgan_size: int = 2048,
        feature_combine_method: str = "difference",
    ):
        super().__init__(
            trainset_filename,
            valset_filename,
            testset_filename,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            num_processes=num_processes,
        )

        self.num_reaction_classes = num_reaction_classes
        self.morgan_radius = morgan_radius
        self.morgan_size = morgan_size
        self.feature_combine_method = feature_combine_method

    def setup(self, stage: Optional[str] = None):

        featurizer = MorganFeaturizer(
            radius=self.morgan_radius,
            size=self.morgan_size,
        )

        self.data_train = USPTOClassicalFeaturesDataset(
            filename=self.trainset_filename,
            featurizer=featurizer,
            feature_type=self.feature_combine_method,
            num_processes=self.num_processes,
        )

        self.data_val = USPTOClassicalFeaturesDataset(
            filename=self.valset_filename,
            featurizer=featurizer,
            feature_type=self.feature_combine_method,
            num_processes=self.num_processes,
        )

        self.data_test = USPTOClassicalFeaturesDataset(
            filename=self.testset_filename,
            featurizer=featurizer,
            feature_type=self.feature_combine_method,
            num_processes=self.num_processes,
        )

        logger.info(
            f"Trainset size: {len(self.data_train)}, valset size: {len(self.data_val)}: "
            f"testset size: {len(self.data_test)}."
        )

    def get_to_model_info(self) -> Dict[str, Any]:

        if self.feature_combine_method == "difference":
            rxn_feats_size = self.morgan_size
        elif self.feature_combine_method == "concatenate":
            rxn_feats_size = 2 * self.morgan_size
        else:
            raise ValueError(
                f"Not supported feature combine method {self.feature_combine_method}"
            )

        d = {
            "reaction_feat_size": rxn_feats_size,
            "num_reaction_classes": self.num_reaction_classes,
        }

        class_weight = self.data_train.get_class_weight(
            num_reaction_classes=self.num_reaction_classes, class_weight_as_1=True
        )
        d["reaction_class_weight"] = class_weight["reaction_type"]

        return d

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            collate_fn=None,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=False,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            collate_fn=None,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            collate_fn=None,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )
