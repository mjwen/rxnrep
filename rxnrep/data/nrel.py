import logging
from typing import Dict, List, Union

import torch

from rxnrep.core.molecule import MoleculeError
from rxnrep.core.reaction import Reaction, ReactionError, smiles_to_reaction
from rxnrep.data.electrolyte import _atom_bond_hop_class_weight_one_bond_break
from rxnrep.data.uspto import USPTODataset

logger = logging.getLogger(__name__)


class NRELDataset(USPTODataset):
    """
    NREL BDE dataset.
    """

    @staticmethod
    def _process_one_reaction_from_input_file(
        smiles_reaction: str, id: str
    ) -> Union[Reaction, None]:
        """
        Helper function to create reactions using multiprocessing.

        Note, not remove H from smiles.
        """

        try:
            reaction = smiles_to_reaction(
                smiles_reaction,
                id=id,
                ignore_reagents=True,
                remove_H=False,
                sanity_check=False,
            )
        except (MoleculeError, ReactionError):
            return None

        return reaction

    def generate_labels(self, normalize: bool = True) -> List[Dict[str, torch.Tensor]]:
        """
        Labels for all reactions.

        Each dict is the labels for one reaction, with keys:
            `atom_hop_dist`, `bond_hop_dist`, `reaction_energy`, `activation_energy`.

        Args:
            normalize: whether to normalize the reaction energy and activation energy
                labels
        """

        # `atom_hop_dist` and `bond_hop_dist` labels
        labels = super().generate_labels()

        # `reaction_energy` and `activation_energy` label

        reaction_energy = torch.as_tensor(
            [rxn.get_property("reaction energy") for rxn in self.reactions],
            dtype=torch.float32,
        )
        if normalize:
            reaction_energy = self.scale_label(reaction_energy, name="reaction_energy")

        # (each energy is a scalar, but here we make it a 1D tensor of 1 element to use
        # the collate_fn, where all energies in a batch is cat to a 1D tensor)
        for re, rxn_label in zip(reaction_energy, labels):
            rxn_label["reaction_energy"] = torch.as_tensor([re], dtype=torch.float32)

        return labels

    def get_class_weight(self) -> Dict[str, torch.Tensor]:
        """
        Create class weight to be used in cross entropy losses.

        Here all the reactions are one bond breaking reactions (A->B and A->B+C),
        and there is no bond creation.
        """
        return _atom_bond_hop_class_weight_one_bond_break(
            self.labels, self.max_hop_distance
        )
