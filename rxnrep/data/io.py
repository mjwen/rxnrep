import multiprocessing
from functools import partial
from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd

from rxnrep.core.molecule import MoleculeError
from rxnrep.core.reaction import Reaction, ReactionError, smiles_to_reaction


def read_smiles_tsv_dataset(
    filename: Path, remove_H: bool, nprocs: int = 1
) -> Tuple[List[Reaction], List[bool]]:
    """
    Read reactions from dataset file.

    Args:
        filename: name of the dataset
        remove_H: whether to remove H from smiles
        nprocs:

    Returns:
        reactions: a list of rxnrep Reaction succeed in converting to dgl graphs.
            The length of this list could be shorter than the number of entries in
            the dataset file (when some entry fails).
        failed: a list of bool indicating whether each entry in the dataset file
            fails or not. The length of the list is the same as the number of
            entries in the dataset file.
    """

    filename = Path(filename).expanduser().resolve()
    df = pd.read_csv(filename, sep="\t")
    smiles_reactions = df["reaction"].tolist()

    ids = [f"{smi}_index-{i}" for i, smi in enumerate(smiles_reactions)]
    if nprocs == 1:
        reactions = [
            smiles_to_reaction_helper(smi, i, remove_H)
            for smi, i in zip(smiles_reactions, ids)
        ]
    else:
        helper = partial(smiles_to_reaction_helper, remove_H=remove_H)
        args = zip(smiles_reactions, ids)
        with multiprocessing.Pool(nprocs) as p:
            reactions = p.starmap(helper, args)

    # column names besides `reaction`
    column_names = df.columns.values.tolist()
    column_names.remove("reaction")

    succeed_reactions = []
    failed = []

    for i, rxn in enumerate(reactions):
        if rxn is None:
            failed.append(True)
        else:
            # keep other info (e.g. label) in input file as reaction property
            for name in column_names:
                rxn.set_property(name, df[name][i])

            succeed_reactions.append(rxn)
            failed.append(False)

    return succeed_reactions, failed


def smiles_to_reaction_helper(
    smiles_reaction: str, id: str, remove_H: bool
) -> Union[Reaction, None]:
    """
    Helper function to create reactions using multiprocessing.

    If fails, return None.
    """

    try:
        reaction = smiles_to_reaction(
            smiles_reaction,
            id=id,
            ignore_reagents=True,
            remove_H=remove_H,
            sanity_check=False,
        )
    except (MoleculeError, ReactionError):
        return None

    return reaction
