from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

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
from rxnrep.data.green import GreenDataset
from rxnrep.data.uspto import USPTODataset
from rxnrep.utils import to_path, yaml_load


def get_prediction(
    model,
    dataset_filename: Path,
    dataset_identifier: str,
    pretrained_path: Path,
    prediction_type: str = "reaction_feature",
) -> List[Dict[str, Any]]:
    """
    Make predictions using a pretrained model.

    Args:
        pretrained_path: path to the directory storing the pretrained model. Two files
            should exists in this directory: 1) checkpoint.ckpt, which is the checkpoint
            file of the lightning model, and 2) dataset_state_dict.yaml, which is the
            state dict of the dataset used in training the model.
        dataset_filename: path to the dataset file to get the predictions for.
        dataset_identifier: Identification of the dataset. Options are `uspto`,
        `electrolyte_full`, electrolyte_two_bond_types`, `green`.
        prediction_type: the type of prediction to return. Options are `reaction_feature`,
            `diff_feature_before_rxn_conv`, and `diff_feature_after_rxn_conv`.
        model: If not `None`, the given model is used; otherwise, load model from the
            given model path.

    Returns:
        Predictions for all data points. Each dict in the list holds the prediction
        for one data point. If the data point (reaction) fails when it was read in the
        dataset, the dict will instead be a `None`, indicating no results for it.
        Example:
            when `prediction_type = "reaction_feature"`, each dict in the returned list
            has the form: {'value': reaction_feature, 'reaction': Reaction}

    """
    pretrained_path = to_path(pretrained_path)
    dataset_state_dict_path = pretrained_path.joinpath("dataset_state_dict.yaml")

    if dataset_identifier == "uspto":
        data_loader = load_uspto_dataset(dataset_state_dict_path, dataset_filename)

    elif dataset_identifier == "electrolyte_full":
        data_loader = load_electrolyte_dataset(
            dataset_state_dict_path, dataset_filename, dataset_type="full"
        )
    elif dataset_identifier == "electrolyte_two_bond_types":
        data_loader = load_electrolyte_dataset(
            dataset_state_dict_path, dataset_filename, dataset_type="two_bond_types"
        )
    elif dataset_identifier == "green":
        data_loader = load_green_dataset(dataset_state_dict_path, dataset_filename)

    else:
        supported = ["uspto", "electrolyte", "electrolyte_two_bond_types", "green"]
        raise PredictionError(
            f"Expect model to be one of {supported}, but got {dataset_identifier}."
        )

    # get evaluation results from the model
    results = evaluate(model, data_loader, prediction_type)

    # reactions
    reactions = data_loader.dataset.reactions

    # split results (the size is different for different reactions)
    if prediction_type in [
        "diff_feature_before_rxn_conv",
        "diff_feature_after_rxn_conv",
    ]:
        num_atoms = [len(rxn.species) for rxn in reactions]
        num_bonds = [
            len(rxn.unchanged_bonds) + len(rxn.lost_bonds) + len(rxn.added_bonds)
            for rxn in reactions
        ]
        results["atom"] = torch.split(results["atom"], num_atoms)
        results["bond"] = torch.split(results["bond"], num_bonds)

    # convert to a list of dict, one for each reaction
    predictions = []
    idx = 0
    for do_fail in data_loader.dataset.get_failed():

        # failed when converting reactions in raw input
        if do_fail:
            predictions.append(None)

        # succeeded reactions
        else:
            # predictions
            d = {k: results[k][idx].numpy() for k in results}

            # add reaction to it
            d["reaction"] = data_loader.dataset.reactions[idx]

            predictions.append(d)
            idx += 1

    return predictions


def evaluate(model, data_loader, prediction_type: str) -> Dict[str, torch.Tensor]:
    """
    Get the model predictions. This is whatever returned by the forward() method of the
    lightning model.

    Args:
        model: lightning model
        data_loader: torch dataloader
        prediction_type: the type of prediction to return. Options are `reaction_feature`,
            `diff_feature_before_rxn_conv`, and `diff_feature_after_rxn_conv.

    Returns:
        Predictions of the model: {prediction_name, prediction_values}.
        For models considered here, prediction values are reaction feature, difference
        feature (depending on prediction_type). It is a 2D tensor (N, D),
        where `N` is the number of data points and `D` is the embedding dimension.
    """

    model.eval()

    results = defaultdict(list)

    with torch.no_grad():
        for batch in data_loader:
            preds = model(batch, returns=prediction_type)

            # preds a dictionary with keys: `atom`, `bond`, and `global`,
            # when prediction_type is `diff_feature_...`
            if isinstance(preds, dict):
                for k, v in preds.items():
                    results[k].append(v.detach())

            # preds is a tensor when prediction_type = reaction_feature;
            elif isinstance(preds, torch.Tensor):
                results["value"].append(preds.detach())

    # make each value a 2D array
    results = {k: torch.cat(v) for k, v in results.items()}

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


def load_green_dataset(state_dict_path: Path, dataset_filename: Path):

    state_dict = yaml_load(state_dict_path)

    dataset = GreenDataset(
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
