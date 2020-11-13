from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch

from rxnrep.prediction.load_model import (
    load_uspto_dataset,
    load_electrolyte_dataset,
    load_model,
    PredictionError,
)
from rxnrep.utils import to_path


def get_prediction(
    model_path: Path, dataset_filename: Path, model_name: str
) -> List[Dict[str, Any]]:
    """
    Make predictions using a pretrained model.

    Args:
        model_path: path to the directory storing the pretrained model.
        dataset_filename: path to the dataset file.
        model_name: name of the model (dataset plus model)

    Returns:
        Predictions for all data points. Each dict in the list holds the prediction
        for one data point. If the data point (reaction) fails when it was read in the
        dataset, the dict will instead be a `None`, indicating no results for it.
    """
    model_path = to_path(model_path)
    model = load_model(model_path)

    if model_name == "uspto":
        data_loader = load_uspto_dataset(model_path, dataset_filename)
    elif model_name == "electrolyte_full":
        data_loader = load_electrolyte_dataset(
            model_path, dataset_filename, dataset_type="full"
        )
    elif model_name == "electrolyte_two_bond_types":
        data_loader = load_electrolyte_dataset(
            model_path, dataset_filename, dataset_type="two_bond_types"
        )
    else:
        supported = ["uspto", "electrolyte", "electrolyte_two_bond_types"]
        raise PredictionError(
            f"Expect model to be one of {supported}, but got {model_name}."
        )

    dataset = data_loader.dataset

    # get embeddings from the model
    embeddings = evaluate(model, data_loader)

    # convert to a list of dict, one for each data point
    predictions = []
    idx = 0
    for do_fail in dataset.get_failed():
        if do_fail:
            # failed when converting reactions in raw input
            predictions.append(None)
        else:
            # succeeded reactions
            d = {k: embeddings[k][idx] for k in embeddings}
            predictions.append(d)
            idx += 1

    return predictions


def evaluate(model, data_loader, device=None) -> Dict[str, np.ndarray]:
    """
    Args:
        model:
        data_loader:
        device:

    Returns:
        {embedding_name: embeddings}. embeddings is a 2D array of shape (N, D),
            where `N` is the number of data points and `D` is the embedding dimension.
            The embedding dimension `D` can be different for different embeddings.
    """

    model.eval()

    nodes = ["atom", "bond", "global"]

    embeddings_before_decoder = []
    embeddings_after_decoder = []

    with torch.no_grad():

        for it, (indices, mol_graphs, rxn_graphs, labels, metadata) in enumerate(
            data_loader
        ):
            mol_graphs = mol_graphs.to(device)
            rxn_graphs = rxn_graphs.to(device)
            feats = {
                nt: mol_graphs.nodes[nt].data.pop("feat").to(device) for nt in nodes
            }

            preds, rxn_embeddings = model(mol_graphs, rxn_graphs, feats, metadata)

            embeddings_before_decoder.append(rxn_embeddings.cpu().numpy())
            embeddings_after_decoder.append(preds["reaction_cluster"].cpu().numpy())

    embeddings = {
        "before_decoder": np.concatenate(embeddings_before_decoder),
        "after_decoder": np.concatenate(embeddings_after_decoder),
    }

    return embeddings
