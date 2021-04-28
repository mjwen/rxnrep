import logging
from collections import Counter

import torch

from rxnrep.data.io import read_mrnet_reaction_dataset
from rxnrep.data.uspto import BaseDatasetWithLabels

logger = logging.getLogger(__name__)


class ElectrolyteDataset(BaseDatasetWithLabels):
    """
    Electrolyte dataset for unsupervised reaction representation.
    """

    def read_file(self, filename):
        logger.info("Start reading dataset file...")

        succeed_reactions, failed = read_mrnet_reaction_dataset(filename, self.nprocs)

        counter = Counter(failed)
        logger.info(
            f"Finish reading reactions. Number succeed {counter[False]}, "
            f"number failed {counter[True]}."
        )

        return succeed_reactions, failed

    def generate_labels(self, normalize: bool = True):
        """
        Labels for all reactions.

        Each dict is the labels for one reaction, with keys:
            `atom_hop_dist`, `bond_hop_dist`, and `reaction_energy`.

        Args:
            normalize: whether to normalize `reaction_energy` labels
        """
        super().generate_labels()

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
