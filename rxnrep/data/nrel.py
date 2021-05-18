import logging
from collections import Counter
from pathlib import Path
from typing import Optional

import torch

from rxnrep.data.datamodule import BaseRegressionDataModule
from rxnrep.data.featurizer import AtomFeaturizer, BondFeaturizer, GlobalFeaturizer
from rxnrep.data.io import read_smiles_tsv_dataset
from rxnrep.data.uspto import BaseLabelledDataset

logger = logging.getLogger(__name__)


class NRELDataset(BaseLabelledDataset):
    """
    NREL BDE dataset.
    """

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
            `atom_hop_dist`, `bond_hop_dist`, `reaction_energy`.

        Args:
            normalize: whether to normalize the reaction energy and activation energy
                labels
        """

        # `reaction_energy` label
        reaction_energy = torch.as_tensor(
            [rxn.get_property("reaction energy") for rxn in self.reactions],
            dtype=torch.float32,
        )
        if normalize:
            reaction_energy = self.scale_label(reaction_energy, name="reaction_energy")

        # (each e is a scalar, but here we make it a 1D tensor of 1 element to use the
        # collate_fn, where all energies in a batch is cat to a 1D tensor)
        for i, e in enumerate(reaction_energy):
            self.labels[i]["reaction_energy"] = torch.as_tensor(
                [e], dtype=torch.float32
            )


class NRELDataModule(BaseRegressionDataModule):
    """
    Electrolyte data module for regression reaction energy and activation energy.
    """

    def setup(self, stage: Optional[str] = None):
        init_state_dict = self.get_init_state_dict()

        atom_featurizer, bond_featurizer, global_featurizer = self._get_featurizers()

        self.data_train = NRELDataset(
            filename=self.trainset_filename,
            atom_featurizer=atom_featurizer,
            bond_featurizer=bond_featurizer,
            global_featurizer=global_featurizer,
            build_reaction_graph=self.build_reaction_graph,
            init_state_dict=init_state_dict,
            num_processes=self.num_processes,
            transform_features=True,
            allow_label_scaler_none=self.allow_label_scaler_none,
        )

        state_dict = self.data_train.state_dict()

        self.data_val = NRELDataset(
            filename=self.valset_filename,
            atom_featurizer=atom_featurizer,
            bond_featurizer=bond_featurizer,
            global_featurizer=global_featurizer,
            build_reaction_graph=self.build_reaction_graph,
            init_state_dict=state_dict,
            num_processes=self.num_processes,
            transform_features=True,
            allow_label_scaler_none=self.allow_label_scaler_none,
        )

        self.data_test = NRELDataset(
            filename=self.testset_filename,
            atom_featurizer=atom_featurizer,
            bond_featurizer=bond_featurizer,
            global_featurizer=global_featurizer,
            build_reaction_graph=self.build_reaction_graph,
            init_state_dict=state_dict,
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
        atom_featurizer = AtomFeaturizer()
        bond_featurizer = BondFeaturizer()
        global_featurizer = GlobalFeaturizer()

        return atom_featurizer, bond_featurizer, global_featurizer
