import logging
from collections import Counter
from typing import Dict

import torch

from rxnrep.data.io import read_mrnet_reaction_dataset
from rxnrep.data.uspto import USPTODataset, get_atom_bond_hop_dist_class_weight

logger = logging.getLogger(__name__)


class ElectrolyteDataset(USPTODataset):
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

        # `reaction_energy` label
        reaction_energy = torch.as_tensor(
            [rxn.get_property("free_energy") for rxn in self.reactions],
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

    def get_class_weight(
        self, only_break_bond: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Create class weight to be used in cross entropy losses.

        Args:
            only_break_bond: whether the dataset only contains breaking bond, i.e.
                does not have lost bond
        """
        return get_atom_bond_hop_dist_class_weight(
            self.labels, self.max_hop_distance, only_break_bond
        )
