"""
Get molecule atom features in different layers as well as the model weight.

This was intended for John's request for preparing some visual for Kristin.
"""
from pathlib import Path

import torch
from monty.serialization import dumpfn

from predict import get_dataloader
from rxnrep.model.regressor import LightningModel
from rxnrep.utils.io import to_path


def evaluate(model, data_loader):
    """
    Return reaction features fo all reactions. A 2D numpy array (N, D) where N
    is the number of reactions and D is the feature dim.
    """
    model.eval()

    with torch.no_grad():
        for batch in data_loader:
            indices, mol_graphs, rxn_graphs, labels, metadata = batch

            feats = {
                nt: mol_graphs.nodes[nt].data.pop("feat") for nt in ["atom", "global"]
            }
            feats["bond"] = mol_graphs.edges["bond"].data.pop("feat")

            mol_feats = model(
                mol_graphs,
                rxn_graphs,
                feats,
                metadata,
                return_mode="molecule_feature",
            )

    # select only atom features
    mol_idx = 0
    atom_feats = {}
    for k, v in mol_feats.items():
        atom_feats[k] = v["atom"][mol_idx].detach().numpy()

    for k, v in atom_feats.items():
        print(k, v.shape)

    dumpfn(atom_feats, "atom_feats.json", indent=2)


def get_model_weights_and_biases(model):
    """
    Get the weights and biases of a model.

    This is very simplified, only get some of them.
    """
    model.eval()

    params = {}
    for name, p in model.named_parameters():

        # if "embedding" in name and "atom" in name:
        #     params["linear"] = p.detach().numpy().T

        if "molecule_conv_layers" in name and "A.mlp.0" in name:
            layer = name.split(".")[2]
            if "weight" in name:
                params[f"mol_conv_layer{layer}.weight"] = p.detach().numpy().T
            if "bias" in name:
                params[f"mol_conv_layer{layer}.bias"] = p.detach().numpy()

    for k, v in params.items():
        print(k, v.shape)

    dumpfn(params, "weights_and_biases.json", indent=2)


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
    evaluate(model, data_loader)

    get_model_weights_and_biases(model)


if __name__ == "__main__":
    model_directory = "./rxnrep_model"
    dataset_path = "~/Downloads/caffine_rxn.json"
    # dataset_path = Path(__file__).parent.joinpath("examples", "reactions_mrnet.json")

    main(model_directory, dataset_path)
