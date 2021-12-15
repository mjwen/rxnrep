"""
Heid regression dataset.

Machine Learning of Reaction Properties via Learned Representations of the
Condensed Graph of Reaction, Esther Heid and William H. Green, JCIM,
https://doi.org/10.1021/acs.jcim.1c00975
"""
import logging
from collections import Counter
from pathlib import Path
from typing import Optional

import torch

from rxnrep.data.datamodule import BaseContrastiveDataModule, BaseRegressionDataModule
from rxnrep.data.dataset import BaseContrastiveDataset, BaseLabelledDataset
from rxnrep.data.featurizer import AtomFeaturizer, BondFeaturizer, GlobalFeaturizer
from rxnrep.data.inout import read_smiles_tsv_dataset

logger = logging.getLogger(__name__)


def read_heid_dataset(filename: Path, remove_H: bool, nprocs: int):
    logger.info("Start reading dataset ...")

    succeed_reactions, failed = read_smiles_tsv_dataset(
        filename, remove_H=remove_H, nprocs=nprocs
    )

    counter = Counter(failed)
    logger.info(
        f"Finish reading dataset. Number succeed {counter[False]}, "
        f"number failed {counter[True]}."
    )

    return succeed_reactions, failed


class HeidDataset(BaseLabelledDataset):
    """
    Green reaction activation energy dataset.
    """

    def read_file(self, filename: Path):
        return read_heid_dataset(filename, self.remove_H, self.nprocs)

    def generate_labels(self, normalize: bool = True):
        """
        Labels for all reactions.

        Args:
            normalize: whether to normalize the regression target
        """

        target = torch.as_tensor(
            [rxn.get_property("target") for rxn in self.reactions],
            dtype=torch.float32,
        )

        if normalize:
            target = self.scale_label(target, name="target")

        # (each target is a scalar, but here we make it a 1D tensor of 1 element to use
        # the collate_fn, where all energies in a batch is cat to a 1D tensor)
        for i, t in enumerate(target):
            self.labels[i].update({"target": torch.as_tensor([t], dtype=torch.float32)})


class HeidContrastiveDataset(BaseContrastiveDataset):
    def read_file(self, filename: Path):
        return read_heid_dataset(filename, self.remove_H, self.nprocs)


class HeidRegressionDataModule(BaseRegressionDataModule):
    """
    Regrression data module for Heid dataset.
    """

    def setup(self, stage: Optional[str] = None):
        init_state_dict = self.get_init_state_dict()

        atom_featurizer, bond_featurizer, global_featurizer = self._get_featurizers()

        self.data_train = HeidDataset(
            filename=self.trainset_filename,
            atom_featurizer=atom_featurizer,
            bond_featurizer=bond_featurizer,
            global_featurizer=global_featurizer,
            build_reaction_graph=self.build_reaction_graph,
            init_state_dict=init_state_dict,
            remove_H=self.remove_H,
            num_processes=self.num_processes,
            transform_features=True,
            allow_label_scaler_none=self.allow_label_scaler_none,
        )

        state_dict = self.data_train.state_dict()

        self.data_val = HeidDataset(
            filename=self.valset_filename,
            atom_featurizer=atom_featurizer,
            bond_featurizer=bond_featurizer,
            global_featurizer=global_featurizer,
            build_reaction_graph=self.build_reaction_graph,
            init_state_dict=state_dict,
            remove_H=self.remove_H,
            num_processes=self.num_processes,
            transform_features=True,
            allow_label_scaler_none=self.allow_label_scaler_none,
        )

        self.data_test = HeidDataset(
            filename=self.testset_filename,
            atom_featurizer=atom_featurizer,
            bond_featurizer=bond_featurizer,
            global_featurizer=global_featurizer,
            build_reaction_graph=self.build_reaction_graph,
            init_state_dict=state_dict,
            remove_H=self.remove_H,
            num_processes=self.num_processes,
            transform_features=True,
            allow_label_scaler_none=self.allow_label_scaler_none,
        )

        # save dataset state dict
        self.data_train.save_state_dict_file(self.state_dict_filename)

        logger.info(
            f"Trainset size: {len(self.data_train)}, valset size: {len(self.data_val)}: "
            f"testset size: {len(self.data_test)}."
        )

    def _get_featurizers(self):

        atom_featurizer_kwargs = {
            "atom_total_degree_one_hot": {"allowable_set": list(range(5))},
            "atom_total_valence_one_hot": {"allowable_set": list(range(5))},
        }

        atom_featurizer = AtomFeaturizer(featurizer_kwargs=atom_featurizer_kwargs)
        bond_featurizer = BondFeaturizer()
        global_featurizer = GlobalFeaturizer()

        return atom_featurizer, bond_featurizer, global_featurizer


class HeidContrastiveDataModule(BaseContrastiveDataModule):
    """
    Heid datamodule for contrastive learning.
    """

    def setup(self, stage: Optional[str] = None):

        init_state_dict = self.get_init_state_dict()

        atom_featurizer_kwargs = {
            "atom_total_degree_one_hot": {"allowable_set": list(range(5))},
            "atom_total_valence_one_hot": {"allowable_set": list(range(5))},
        }

        atom_featurizer = AtomFeaturizer(featurizer_kwargs=atom_featurizer_kwargs)
        bond_featurizer = BondFeaturizer()
        global_featurizer = GlobalFeaturizer()

        self.data_train = HeidContrastiveDataset(
            filename=self.trainset_filename,
            atom_featurizer=atom_featurizer,
            bond_featurizer=bond_featurizer,
            global_featurizer=global_featurizer,
            build_reaction_graph=self.build_reaction_graph,
            init_state_dict=init_state_dict,
            remove_H=self.remove_H,
            num_processes=self.num_processes,
            transform_features=True,
            transform1=self.transform1,
            transform2=self.transform2,
        )

        state_dict = self.data_train.state_dict()

        self.data_val = HeidContrastiveDataset(
            filename=self.valset_filename,
            atom_featurizer=atom_featurizer,
            bond_featurizer=bond_featurizer,
            global_featurizer=global_featurizer,
            build_reaction_graph=self.build_reaction_graph,
            init_state_dict=state_dict,
            remove_H=self.remove_H,
            num_processes=self.num_processes,
            transform_features=True,
            transform1=self.transform1,
            transform2=self.transform2,
        )

        self.data_test = HeidContrastiveDataset(
            filename=self.testset_filename,
            atom_featurizer=atom_featurizer,
            bond_featurizer=bond_featurizer,
            global_featurizer=global_featurizer,
            build_reaction_graph=self.build_reaction_graph,
            init_state_dict=state_dict,
            remove_H=self.remove_H,
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
