import logging
import multiprocessing
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd
import torch

from rxnrep.core.molecule import MoleculeError
from rxnrep.core.reaction import Reaction, ReactionError, smiles_to_reaction
from rxnrep.data.uspto import USPTODataset
from rxnrep.utils import to_path

logger = logging.getLogger(__name__)


class GreenDataset(USPTODataset):
    """
    Green reaction activation energy dataset.
    """

    @staticmethod
    def read_file(filename: Path, nprocs: int):

        # read file
        logger.info("Start reading dataset file...")

        filename = to_path(filename)
        df = pd.read_csv(filename, sep="\t")
        smiles_reactions = df["reaction"].tolist()
        activation_energy = df["activation energy"].to_list()
        reaction_energy = df["reaction enthalpy"].to_list()

        logger.info("Finish reading dataset file...")

        # convert to reactions and labels
        logger.info("Start converting to reactions...")

        ids = [f"{smi}_index-{i}" for i, smi in enumerate(smiles_reactions)]
        if nprocs == 1:
            reactions = [
                process_one_reaction_from_input_file(smi, i)
                for smi, i in zip(smiles_reactions, ids)
            ]
        else:
            args = zip(smiles_reactions, ids)
            with multiprocessing.Pool(nprocs) as p:
                reactions = p.starmap(process_one_reaction_from_input_file, args)

        failed = []
        succeed_reactions = []
        succeed_labels = defaultdict(list)

        for i, rxn in enumerate(reactions):
            if rxn is None:
                failed.append(True)
            else:
                failed.append(False)
                # TODO, remove succeed_labels, because labels set as reaction property
                succeed_labels["activation_energy"].append(activation_energy[i])
                succeed_labels["reaction_energy"].append(reaction_energy[i])
                rxn.set_property("activation_energy", activation_energy[i])
                rxn.set_property("reaction_energy", reaction_energy[i])
                succeed_reactions.append(rxn)

        counter = Counter(failed)
        logger.info(
            f"Finish converting to reactions. Number succeed {counter[False]}, "
            f"number failed {counter[True]}."
        )

        return succeed_reactions, succeed_labels, failed

    def generate_labels(self) -> List[Dict[str, torch.Tensor]]:
        """
        Labels for all reactions.

        Each dict is the labels for one reaction, with keys:
            `atom_hop_dist`, `bond_hop_dist`, `reaction_energy`, `activation_energy`.
        """

        # `atom_hop_dist` and `bond_hop_dist` labels
        labels = super(GreenDataset, self).generate_labels()

        # `reaction_energy` and `activation_energy` label
        # (each is a scalar, but here we make it a 1D tensor of 1 element to use the
        # collate_fn, where all energies in a batch is cat to a 1D tensor)
        reaction_energy = self._raw_labels["reaction_energy"]
        activation_energy = self._raw_labels["activation_energy"]
        for re, ae, rxn_label in zip(reaction_energy, activation_energy, labels):
            rxn_label["reaction_energy"] = torch.as_tensor([re], dtype=torch.float32)
            rxn_label["activation_energy"] = torch.as_tensor([ae], dtype=torch.float32)

        return labels


def process_one_reaction_from_input_file(
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
