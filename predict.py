"""
BonDNet prediction script.
"""
import json
import subprocess
from pathlib import Path
from typing import List

import click
import numpy as np
import torch
from rdkit import RDLogger
from torch.utils.data import DataLoader

from rxnrep.data.electrolyte import ElectrolyteDataset
from rxnrep.data.featurizer import (
    AtomFeaturizerMinimum2,
    BondFeaturizerMinimum,
    GlobalFeaturizer,
)
from rxnrep.model.regressor import LightningModel
from rxnrep.utils.google_drive import download_model
from rxnrep.utils.io import to_path, yaml_load

RDLogger.logger().setLevel(RDLogger.CRITICAL)

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


def get_dataloader(state_dict_path: Path, dataset_filename: Path):
    """
    NOTE, the featurizer should be the same as the ones in training datamodule.
    """

    state_dict = yaml_load(state_dict_path)
    allowable_charge = [-1, 0, 1]

    atom_featurizer = AtomFeaturizerMinimum2()
    bond_featurizer = BondFeaturizerMinimum()
    global_featurizer = GlobalFeaturizer(allowable_charge=allowable_charge)

    dataset = ElectrolyteDataset(
        filename=dataset_filename,
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer,
        global_featurizer=global_featurizer,
        transform_features=True,
        init_state_dict=state_dict_path,
    )

    # check species and charge
    _check_species(dataset.get_species(), state_dict["species"])
    # _check_charge(dataset.get_charges(), supported_charges=allowable_charge)

    # create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=200,
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
        raise DatasetError(
            f"Model trained with a dataset having species: {supported}; Cannot make "
            f"predictions for molecule containing species: {not_supported}."
        )


def _check_charge(charges: List[int], supported_charges: List[int]):
    not_supported = set(charges) - set(supported_charges)
    if not_supported:
        not_supported = ", ".join(sorted([str(i) for i in not_supported]))
        supported = ", ".join([str(i) for i in supported_charges])
        raise DatasetError(
            f"Model trained with a dataset of molecules having charges: {supported}; "
            f"Cannot make predictions for molecule containing charges: {not_supported}."
        )


def evaluate(model, data_loader):
    """
    Return reaction energy fo all reactions. A 1D array of lengh N, where N is the
    total number of reactions.
    """
    model.eval()

    all_preds = []

    with torch.no_grad():
        for batch in data_loader:
            indices, mol_graphs, rxn_graphs, labels, metadata = batch

            feats = {
                nt: mol_graphs.nodes[nt].data.pop("feat") for nt in ["atom", "global"]
            }
            feats["bond"] = mol_graphs.edges["bond"].data.pop("feat")

            pred = model(
                mol_graphs,
                rxn_graphs,
                feats,
                metadata,
                return_mode="reaction_energy",
            )

            all_preds.append(pred.numpy())

    all_preds = np.concatenate(all_preds)

    return all_preds


def main(model_directory: Path, data_filename: Path):
    """
    Main predict function.

    This will write a json file, with the same content in the input `data_filename` an
    additional key 'predicted_reaction_energy' written for each entry.

    Args:
        model_directory: path to the directory that stores the trained model. Should
            contain three files:
            1. checkpoint.ckpt
            2. dataset_state_dict.yaml
            3. hydra_final.yaml (this is not used in recovery, but we keep it to
            describe the model
        data_filename: path to a json file containing the data

    """
    ckpt_path = to_path(model_directory).joinpath("checkpoint.ckpt")
    model = LightningModel.load_from_checkpoint(ckpt_path, map_location="cpu")

    dataset_state_dict = to_path(model_directory).joinpath("dataset_state_dict.yaml")
    data_loader = get_dataloader(dataset_state_dict, data_filename)
    preds = evaluate(model, data_loader)

    # write results
    with open(data_filename, "r") as f:
        data = json.load(f)

    assert len(data) == len(preds), (
        f"Something fishy happens, number of predictions {len(preds)} is not equal to "
        f"the number of input data points {len(data)}"
    )

    for d, p in zip(data, preds):
        d["predicted_reaction_energy"] = float(p)

    with open("results.json", "w") as f:
        json.dump(data, f)


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument("data_filename", type=str)
@click.option("--model", type=click.Path(exists=True))
def cli(data_filename, model):

    if model is None:
        model = Path.cwd().joinpath("rxnrep_model")
        if model.exists():
            print("\n\nFind model directory `./rxnrep_model`; will reuse it.")
        else:
            file_id = "13SS3DMf7CSPMKSimBi_S84AHuv6jVmoi"
            date = "20210630"
            download_model(file_id, date, directory=model)

    main(model, data_filename)

    print("Finish prediction. Results written to `result.json`.")


class DatasetError(Exception):
    def __init__(self, msg=None):
        super(DatasetError, self).__init__(msg)
        self.msg = msg


if __name__ == "__main__":

    dataset_path = Path(__file__).parent.joinpath("examples", "reactions_mrnet.json")
    subprocess.call(["bondnet", dataset_path])
