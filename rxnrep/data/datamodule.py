import logging
import os
from typing import Any, Dict, Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from rxnrep.data.featurizer import AtomFeaturizer, BondFeaturizer, GlobalFeaturizer
from rxnrep.data.uspto import USPTODataset
from rxnrep.utils.io import to_path

logger = logging.getLogger(__file__)


class UsptoDataModule(LightningDataModule):
    """

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data
    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html

    Args:
        num_reaction_classes: number of reaction class of the dataset. `None` means the
        dataset has no reaction type label.
    """

    def __init__(
        self,
        trainset_filename,
        valset_filename,
        testset_filename,
        batch_size: int = 100,
        num_workers: int = 0,
        pin_memory: bool = True,
        num_processes: int = 1,
        restore: bool = False,
        state_dict_filename: Optional[os.PathLike] = None,
        pretrained_model_state_dict_filename: Optional[os.PathLike] = None,
        build_reaction_graph: bool = True,
        num_reaction_classes: Optional[int] = None,
    ):
        super().__init__()

        self.trainset_filename = trainset_filename
        self.valset_filename = valset_filename
        self.testset_filename = testset_filename

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.num_processes = num_processes

        self.restore = restore
        self.state_dict_filename = state_dict_filename
        self.pretrained_model_state_dict_filename = pretrained_model_state_dict_filename

        self.build_reaction_graph = build_reaction_graph
        self.num_reaction_classes = num_reaction_classes

        # self.dims is returned when you call datamodule.size()
        self.dims = (1, 28, 28)

        self.data_train = None
        self.data_val = None
        self.data_test = None

        self.initialized = False

    # def prepare_data(self):
    #     """Download data if needed. This method is called only from a single GPU.
    #     Do not use it to assign state (self.x = y)."""
    #     MNIST(self.data_dir, train=True, download=True)
    #     MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        """
        Load data.

        Set variables: self.data_train, self.data_val, self.data_test.
        """

        init_state_dict = self.determine_init_state_dict()

        has_class_label = self.num_reaction_classes is not None

        self.data_train = USPTODataset(
            filename=self.trainset_filename,
            atom_featurizer=AtomFeaturizer(),
            bond_featurizer=BondFeaturizer(),
            global_featurizer=GlobalFeaturizer(),
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
            atom_featurizer=AtomFeaturizer(),
            bond_featurizer=BondFeaturizer(),
            global_featurizer=GlobalFeaturizer(),
            build_reaction_graph=self.build_reaction_graph,
            init_state_dict=state_dict,
            num_processes=self.num_processes,
            transform_features=True,
            # label args
            has_class_label=has_class_label,
        )

        self.data_test = USPTODataset(
            filename=self.testset_filename,
            atom_featurizer=AtomFeaturizer(),
            bond_featurizer=BondFeaturizer(),
            global_featurizer=GlobalFeaturizer(),
            build_reaction_graph=self.build_reaction_graph,
            init_state_dict=state_dict,
            num_processes=self.num_processes,
            transform_features=True,
            # label args
            has_class_label=has_class_label,
        )

        # save dataset state dict
        self.data_train.save_state_dict_file(self.state_dict_filename)

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
            shuffle=True,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            collate_fn=self.data_test.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=False,
        )

    def get_to_model_info(self) -> Dict[str, Any]:
        """
        Pack necessary dataset info as as dict, and this dict will be passed as
        arguments to the model.
        """
        d = {"feature_size": self.data_train.feature_size}

        if self.num_reaction_classes:
            d["num_reaction_classes"] = self.num_reaction_classes

            class_weight = self.data_train.get_class_weight(
                num_reaction_classes=self.num_reaction_classes, class_weight_as_1=True
            )
            d["reaction_class_weight"] = class_weight["reaction_type"]

        return d

    def determine_init_state_dict(self):
        """
        Determine the value of dataset state dict based on:
        - whether this is in finetune model based on pretrained_model_state_dict_filename
        - restore
        """

        # finetune mode
        if self.pretrained_model_state_dict_filename:

            if not to_path(self.pretrained_model_state_dict_filename).exists():
                raise FileNotFoundError(
                    f"Cannot find pretrained model state dict file at: "
                    f"{self.pretrained_model_state_dict_filename}"
                )
            else:
                init_state_dict = self.pretrained_model_state_dict_filename

        # regular training mode
        else:
            if self.restore:

                if self.state_dict_filename is None:
                    logger.warning(
                        "Restore training but `state_dict_filename=None`. Dataset "
                        "statistics (e.g. feature mean and standard deviation) will be "
                        "computed from trainset."
                    )
                    init_state_dict = None

                elif not to_path(self.state_dict_filename).exists():
                    # we throw a warning instead of error because when training on HPC,
                    # it may fail before data dict is generated.
                    logger.warning(
                        f"Restore training with "
                        f"`state_dict_filename={self.state_dict_filename}`. "
                        f"But cannot find this file. Setting it to `None` to compute "
                        f"dataset statistics (e.g. feature mean and standard deviation) "
                        "from the trainset."
                    )
                    init_state_dict = None

                else:
                    init_state_dict = self.state_dict_filename

            else:
                init_state_dict = None

        return init_state_dict
