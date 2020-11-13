import yaml
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader

from rxnrep.data.uspto import USPTODataset
from rxnrep.data.electrolyte import ElectrolyteDataset, ElectrolyteDatasetTwoBondType
from rxnrep.data.featurizer import (
    AtomFeaturizer,
    AtomFeaturizerMinimum,
    BondFeaturizer,
    BondFeaturizerMinimum,
    GlobalFeaturizer,
)
from rxnrep.model.model import ReactionRepresentation
from rxnrep.scripts.utils import load_checkpoints
from rxnrep.utils import yaml_load


def load_model(model_path: Path):
    """
    Load a pretrained model.
    """

    # Cannot use rxnrep.utils.yaml_load, which uses the safe_loader.
    # see: https://github.com/yaml/pyyaml/issues/266
    with open(model_path.joinpath("train_args.yaml"), "r") as f:
        args = yaml.load(f, Loader=yaml.Loader)

    ### model
    model = ReactionRepresentation(
        in_feats=args.feature_size,
        embedding_size=args.embedding_size,
        # encoder
        molecule_conv_layer_sizes=args.molecule_conv_layer_sizes,
        molecule_num_fc_layers=args.molecule_num_fc_layers,
        molecule_batch_norm=args.molecule_batch_norm,
        molecule_activation=args.molecule_activation,
        molecule_residual=args.molecule_residual,
        molecule_dropout=args.molecule_dropout,
        reaction_conv_layer_sizes=args.reaction_conv_layer_sizes,
        reaction_num_fc_layers=args.reaction_num_fc_layers,
        reaction_batch_norm=args.reaction_batch_norm,
        reaction_activation=args.reaction_activation,
        reaction_residual=args.reaction_residual,
        reaction_dropout=args.reaction_dropout,
        # bond type decoder
        bond_type_decoder_hidden_layer_sizes=args.node_decoder_hidden_layer_sizes,
        bond_type_decoder_activation=args.node_decoder_activation,
        # atom in reaction center decoder
        atom_in_reaction_center_decoder_hidden_layer_sizes=args.node_decoder_hidden_layer_sizes,
        atom_in_reaction_center_decoder_activation=args.node_decoder_activation,
        # clustering decoder
        reaction_cluster_decoder_hidden_layer_sizes=args.cluster_decoder_hidden_layer_sizes,
        reaction_cluster_decoder_activation=args.cluster_decoder_activation,
        reaction_cluster_decoder_output_size=args.cluster_decoder_projection_head_size,
        ##
        bond_type_decoder_num_classes=args.bond_type_decoder_num_classes,
    )

    load_checkpoints(
        {"model": model},
        map_location=torch.device("cpu"),
        filename=model_path.joinpath("checkpoint.pkl"),
    )

    return model


def load_uspto_dataset(model_path: Path, dataset_filename: Path):

    state_dict_filename = model_path.joinpath("dataset_state_dict.yaml")
    state_dict = yaml_load(state_dict_filename)

    dataset = USPTODataset(
        filename=dataset_filename,
        atom_featurizer=AtomFeaturizer(),
        bond_featurizer=BondFeaturizer(),
        global_featurizer=GlobalFeaturizer(),
        transform_features=True,
        init_state_dict=state_dict_filename,
    )

    # check species and charge
    _check_species(dataset.get_species(), state_dict["species"])
    # _check_charge(dataset.get_charges(), state_dict["charges"])

    # create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=100,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        drop_last=False,
    )

    return data_loader


def load_electrolyte_dataset(
    model_path: Path, dataset_filename: Path, dataset_type: str = "full"
):
    """
    Args:
        model_path:
        dataset_filename:
        dataset_type: ['full' | 'two_bond_types']. The type of the dataset. `full` means
            the decoder will use three bond types (unchanged, lost, or added).
            `two_bond_types` means the decoder will use two bond types (unchanged or
            changed).
    """

    state_dict_filename = model_path.joinpath("dataset_state_dict.yaml")
    state_dict = yaml_load(state_dict_filename)
    allowable_charge = [-1, 0, 1]

    if dataset_type == "full":
        DT = ElectrolyteDataset
    elif dataset_type == "two_bond_types":
        DT = ElectrolyteDatasetTwoBondType
    else:
        raise PredictionError(f"Unsupported dataset type {dataset_type}")

    dataset = DT(
        filename=dataset_filename,
        atom_featurizer=AtomFeaturizerMinimum(),
        bond_featurizer=BondFeaturizerMinimum(),
        global_featurizer=GlobalFeaturizer(allowable_charge=allowable_charge),
        transform_features=True,
        init_state_dict=state_dict_filename,
    )

    # check species and charge
    _check_species(dataset.get_species(), state_dict["species"])
    _check_charge(dataset.get_charges(), supported_charges=allowable_charge)

    # create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=100,
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
