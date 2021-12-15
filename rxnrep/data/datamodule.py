import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from rxnrep.utils.io import to_path

logger = logging.getLogger(__file__)


class BaseDataModule(LightningDataModule):
    """
    Base datamodule.

    Args:
        trainset_filename: path to the training set file
        valset_filename: path to the validation set file
        testset_filename: path to the validation set file
        batch_size: batch size per process
        num_workers: number of processes for dataloader
        pin_memory: pin gpu memory
        num_processes: number of processes to process data from file
        state_dict_filename: path to save the state dict of the data module.
        restore_state_dict_filename: If not `None`, the model is running in
            restore mode and the initial state dict is read from this file. If `None`,
            the model in running in regular mode and this is ignored.
            Note the difference between this and `state_dict_filename`.
            `state_dict_filename` only specifies the output state dict, does not care
            about how the initial state dict is obtained: it could be restored from
            `restore_state_dict_file` or computed from the dataset.
            pretrained model used in finetune?
        build_reaction_graph: whether to build reaction graph from reactants and products
        remove_H: whether to remove H from graph
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
        state_dict_filename: Union[str, Path] = "dataset_state_dict.yaml",
        restore_state_dict_filename: Optional[Union[str, Path]] = None,
        build_reaction_graph: bool = True,
        remove_H: bool = True,
    ):
        super().__init__()

        self.trainset_filename = trainset_filename
        self.valset_filename = valset_filename
        self.testset_filename = testset_filename

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.num_processes = num_processes

        self.state_dict_filename = state_dict_filename
        self.restore_state_dict_filename = restore_state_dict_filename

        self.build_reaction_graph = build_reaction_graph
        self.remove_H = remove_H

        self.data_train = None
        self.data_val = None
        self.data_test = None

    def prepare_data(self):
        """
        Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """
        Load data.

        Set variables: self.data_train, self.data_val, self.data_test.
        """

        # init_state_dict = self.get_init_state_dict()
        #
        # self.data_train = ...
        # self.data_val = ...
        # self.data_test = ...
        #
        # # save dataset state dict
        # self.data_train.save_state_dict_file(self.state_dict_filename)

        raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            collate_fn=self.data_train.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=False,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            collate_fn=self.data_val.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            collate_fn=self.data_test.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )

    def get_to_model_info(self) -> Dict[str, Any]:
        """
        Pack necessary dataset info as as dict, and this dict will be passed as
        arguments to the model. Such info might include feature size, number of classes
        for reaction.
        """
        raise NotImplementedError

    def get_init_state_dict(self):
        """
        Determine the value of dataset state dict based on:
        - whether this is in finetune model based on pretrained_model_state_dict_filename
        - restore_state_dict_filename
        """

        # restore training
        if self.restore_state_dict_filename:
            init_state_dict = to_path(self.restore_state_dict_filename)

            if not init_state_dict.exists():
                raise FileNotFoundError(
                    "Cannot restore datamodule. Dataset state dict file does not "
                    f"exist: {init_state_dict}"
                )

        # regular training mode
        else:
            init_state_dict = None

        return init_state_dict


class BaseClassificationDataModule(BaseDataModule):
    """
    Base classification datamodule.

    Args:
        num_reaction_classes: number of reaction class of the dataset.
    """

    def __init__(
        self,
        trainset_filename: Union[str, Path],
        valset_filename: Union[str, Path],
        testset_filename: Union[str, Path],
        *,
        num_reaction_classes: int,
        state_dict_filename: Union[str, Path] = "dataset_state_dict.yaml",
        restore_state_dict_filename: Optional[Union[str, Path]] = None,
        batch_size: int = 100,
        num_workers: int = 0,
        pin_memory: bool = True,
        num_processes: int = 1,
        build_reaction_graph: bool = True,
        remove_H: bool = True,
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
            remove_H=remove_H,
        )

        self.num_reaction_classes = num_reaction_classes

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


class BaseRegressionDataModule(BaseDataModule):
    """
    Base regression datamodule.
    Args:
        allow_label_scaler_none: when `pretrained_state_dict_filename` is provided,
            whether to allow label_scaler is None in state dict. If `False`, will raise
            exception. If `True`, will recompute label mean and std. This is mainly used
            for the pretrain-finetune case, where the pretrain model does not use labels
            (e.g. contrastive training). For restoring a regular regression training,
            should set this to `False`.
    """

    def __init__(
        self,
        trainset_filename: Union[str, Path],
        valset_filename: Union[str, Path],
        testset_filename: Union[str, Path],
        *,
        allow_label_scaler_none: bool = False,
        state_dict_filename: Union[str, Path] = "dataset_state_dict.yaml",
        restore_state_dict_filename: Optional[Union[str, Path]] = None,
        batch_size: int = 100,
        num_workers: int = 0,
        pin_memory: bool = True,
        num_processes: int = 1,
        build_reaction_graph: bool = True,
        remove_H: bool = True,
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
            remove_H=remove_H,
        )

        self.allow_label_scaler_none = allow_label_scaler_none

    def get_to_model_info(self) -> Dict[str, Any]:
        d = {
            "feature_size": self.data_train.feature_size,
            "label_scaler": self.data_train.get_label_scaler(),
        }

        return d


class BaseContrastiveDataModule(BaseDataModule):
    """
    Base datamodule for contrastive learning.

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
        remove_H: bool = True,
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
            remove_H=remove_H,
        )
        self.transform1 = transform1
        self.transform2 = transform2

    def get_to_model_info(self) -> Dict[str, Any]:
        d = {"feature_size": self.data_train.feature_size}

        return d


class BaseMorganDataModule(BaseDataModule):
    """
    Base datamodule using Morgan feats for classification.

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
        num_reaction_classes: int,
        morgan_radius: int = 2,
        morgan_size: int = 2048,
        feature_combine_method: str = "difference",
        batch_size: int = 100,
        num_workers: int = 0,
        pin_memory: bool = True,
        num_processes: int = 1,
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

    def get_to_model_info(self) -> Dict[str, Any]:

        if self.feature_combine_method == "difference":
            rxn_feats_size = self.morgan_size
        elif self.feature_combine_method == "concatenate":
            rxn_feats_size = 2 * self.morgan_size
        else:
            raise ValueError(
                f"Not supported feature combine method {self.feature_combine_method}"
            )

        class_weight = self.data_train.get_class_weight(
            num_reaction_classes=self.num_reaction_classes, class_weight_as_1=True
        )

        d = {
            "reaction_feat_size": rxn_feats_size,
            "num_reaction_classes": self.num_reaction_classes,
            "reaction_class_weight": class_weight["reaction_type"],
        }

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
