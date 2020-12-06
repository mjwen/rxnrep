from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from rxnrep.data.electrolyte import ElectrolyteDataset, ElectrolyteDatasetNoAddedBond
from rxnrep.data.featurizer import (
    AtomFeaturizer,
    AtomFeaturizerMinimum,
    BondFeaturizer,
    BondFeaturizerMinimum,
    GlobalFeaturizer,
)
from rxnrep.data.uspto import USPTODataset
from rxnrep.scripts.train_electrolyte import LightningModel as ModelElectrolyteFull
from rxnrep.scripts.train_electrolyte_no_added_bond import (
    LightningModel as ModelElectrolyteTwoBondType,
)
from rxnrep.scripts.train_uspto import LightningModel as ModelUspto
from rxnrep.utils import to_path, yaml_load


def get_prediction(
    model_path: Path, dataset_filename: Path, model_name: str
) -> List[Dict[str, Any]]:
    """
    Make predictions using a pretrained model.

    Args:
        model_path: path to the directory storing the pretrained model. Two files
            should exists in this directory: 1) checkpoint.ckpt, which is the checkpoint
            file of the lightning model, and 2) dataset_state_dict.yaml, which is the
            state dict of the dataset used in training the model.
        dataset_filename: path to the dataset file to get the predictions for.
        model_name: name of the model (dataset plus model)

    Returns:
        Predictions for all data points. Each dict in the list holds the prediction
        for one data point. If the data point (reaction) fails when it was read in the
        dataset, the dict will instead be a `None`, indicating no results for it.
    """
    model_path = to_path(model_path)
    ckpt_path = model_path.joinpath("checkpoint.ckpt")
    dataset_state_dict_path = model_path.joinpath("dataset_state_dict.yaml")

    if model_name == "uspto":
        model = ModelUspto.load_from_checkpoint(
            str(ckpt_path), map_location=torch.device("cpu")
        )
        data_loader = load_uspto_dataset(dataset_state_dict_path, dataset_filename)

    elif model_name == "electrolyte_full":
        model = ModelElectrolyteFull.load_from_checkpoint(
            str(ckpt_path), map_location=torch.device("cpu")
        )
        data_loader = load_electrolyte_dataset(
            dataset_state_dict_path, dataset_filename, dataset_type="full"
        )
    elif model_name == "electrolyte_two_bond_types":
        model = ModelElectrolyteTwoBondType.load_from_checkpoint(
            str(ckpt_path), map_location=torch.device("cpu")
        )
        data_loader = load_electrolyte_dataset(
            dataset_state_dict_path, dataset_filename, dataset_type="two_bond_types"
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


def evaluate(model, data_loader) -> Dict[str, np.ndarray]:
    """
    Get the model predictions. This is whatever returned by the forward() method of the
    lightning model.

    Args:
        model: lightning model
        data_loader: torch dataloader

    Returns:
        Predictions of the model: {prediction_name, prediction_values}. For models
        considered here, prediction values are reaction  embedding, a 2D array of
        shape (N, D), where `N` is the number of data points and `D` is the embedding
        dimension.
    """

    model.eval()

    embeddings = []

    with torch.no_grad():
        for batch in data_loader:
            preds = model(batch)
            embeddings.append(preds.cpu().numpy())

        results = {"embeddings": np.concatenate(embeddings)}

    return results


def load_uspto_dataset(state_dict_path: Path, dataset_filename: Path):

    state_dict = yaml_load(state_dict_path)

    dataset = USPTODataset(
        filename=dataset_filename,
        atom_featurizer=AtomFeaturizer(),
        bond_featurizer=BondFeaturizer(),
        global_featurizer=GlobalFeaturizer(),
        transform_features=True,
        init_state_dict=state_dict_path,
    )

    # check species and charge
    _check_species(dataset.get_species(), state_dict["species"])
    # _check_charge(dataset.get_charges(), state_dict["charges"])

    # create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=1000,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        drop_last=False,
    )

    return data_loader


def load_electrolyte_dataset(
    state_dict_path: Path, dataset_filename: Path, dataset_type: str = "full"
):
    """
    Args:
        state_dict_path:
        dataset_filename:
        dataset_type: ['full' | 'two_bond_types']. The type of the dataset. `full` means
            the decoder will use three bond types (unchanged, lost, or added).
            `two_bond_types` means the decoder will use two bond types (unchanged or
            changed).
    """

    state_dict = yaml_load(state_dict_path)
    allowable_charge = [-1, 0, 1]

    if dataset_type == "full":
        DT = ElectrolyteDataset
    elif dataset_type == "two_bond_types":
        DT = ElectrolyteDatasetNoAddedBond
    else:
        raise PredictionError(f"Unsupported dataset type {dataset_type}")

    dataset = DT(
        filename=dataset_filename,
        atom_featurizer=AtomFeaturizerMinimum(),
        bond_featurizer=BondFeaturizerMinimum(),
        global_featurizer=GlobalFeaturizer(allowable_charge=allowable_charge),
        transform_features=True,
        init_state_dict=state_dict_path,
    )

    # check species and charge
    _check_species(dataset.get_species(), state_dict["species"])
    _check_charge(dataset.get_charges(), supported_charges=allowable_charge)

    # create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=1000,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        drop_last=False,
    )

    return data_loader


def _check_species(species: List[str], supported_species: List[str]):
    not_supported = set(species) - set(supported_species)
    if not_supported:
        not_supported = ", ".join(sorted(not_supported))
        supported = ", ".join(supported_species)
        raise PredictionError(
            f"Model trained with a dataset having species: {supported}; Cannot make "
            f"predictions for molecule containing species: {not_supported}."
        )


def _check_charge(charges: List[int], supported_charges: List[int]):
    not_supported = set(charges) - set(supported_charges)
    if not_supported:
        not_supported = ", ".join(sorted([str(i) for i in not_supported]))
        supported = ", ".join([str(i) for i in supported_charges])
        raise PredictionError(
            f"Model trained with a dataset of molecules haaving charges: {supported}; "
            f"Cannot make predictions for molecule containing charges: {not_supported}."
        )


class PredictionError(Exception):
    def __init__(self, msg=None):
        super(PredictionError, self).__init__(msg)
        self.msg = msg
