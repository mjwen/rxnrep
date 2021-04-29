import logging
from collections import Counter
from pathlib import Path

import torch

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
        super().generate_labels()

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
