import tempfile
from pathlib import Path
from typing import List, Union

import pandas as pd
import torch
from torch.utils.data import DataLoader

from rxnrep.data.featurizer import AtomFeaturizer, BondFeaturizer, GlobalFeaturizer
from rxnrep.data.uspto import USPTODataset
from rxnrep.model.pretrainer import LightningModel
from rxnrep.utils.io import yaml_load


def load_pretrained_model(checkpoint: Union[str, Path], map_location: str = "cpu"):
    """
    Load the pretrained model using Schneider or TPL100 dataset.
    """

    model = LightningModel.load_from_checkpoint(
        checkpoint_path=str(checkpoint), map_location=map_location
    )

    return model


def load_uspto_dataset(
    smiles_reactions: List[str], state_dict_path: Path, batch_size=100
):

    # USPTODataset reads reactions from a file, here the smiles reactions are from a
    # list, we create a
    with tempfile.TemporaryDirectory() as dirpath:
        filename = Path(dirpath).joinpath("smiles_rxn.tsv")
        df = pd.DataFrame({"reaction": smiles_reactions})
        df.to_csv(filename, sep="\t")

        dataset = USPTODataset(
            filename=filename,
            atom_featurizer=AtomFeaturizer(),
            bond_featurizer=BondFeaturizer(),
            global_featurizer=GlobalFeaturizer(),
            transform_features=True,
            init_state_dict=state_dict_path,
            has_class_label=False,
        )

    state_dict = yaml_load(state_dict_path)
    _check_species(dataset.get_species(), state_dict["species"])
    # _check_charge(dataset.get_charges(), state_dict["charges"])

    # check on fails
    failed = dataset.get_failed()
    if any(failed):
        indices = [i for i, x in enumerate(failed) if x]
        raise RuntimeError(
            f"Cannot make predictions for reactions {indices}, because it fails to "
            "convert their corresponding reaction SMILES to graphs. Currently, "
            "a SMILES reaction should be balanced and the atoms in the reactants and "
            "products should be mapped."
        )

    # create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        drop_last=False,
    )

    return data_loader


def get_rxnrep_fingerprint(
    smiles_reactions: List[str],
    pretrain_dataset: str = "Schneider",
    mode: str = "eval",
    device: str = "cpu",
) -> torch.Tensor:
    """

    Args:
        smiles_reactions: A list of smiles reactions to get their RxnRep fingerprints.
        pretrain_dataset: name of the pretraining dataset, `Schneider` or `TPL100`.
        mode: whether to run in `eval` mode or `train` model? In `eval` mode,
            no gradients will be retrained. In `train` mode, the gradients will be
            retrained, and then you can continue fine-tuning the model.
        device: where to run the model.

    Returns: Tensors of shape (N, D). Each row is the fingerprint of an input
        reaction, where N is the total number of reactions and D is the dimension of
        the fingerprints.
    """

    def get_predictions(model, loader):

        all_reaction_feats = []
        for batch in loader:
            indices, mol_graphs, rxn_graphs, labels, metadata = batch

            feats = {
                nt: mol_graphs.nodes[nt].data.pop("feat") for nt in ["atom", "global"]
            }
            feats["bond"] = mol_graphs.edges["bond"].data.pop("feat")

            _, rxn_feats = model(mol_graphs, rxn_graphs, feats, metadata)

            all_reaction_feats.append(rxn_feats)

        all_reaction_feats = torch.cat(all_reaction_feats)

        return all_reaction_feats

    supported = ["Schneider", "TPL100"]

    if pretrain_dataset.lower() not in [s.lower() for s in supported]:
        raise ValueError(
            f"Supported pretrained dataset includes {supported}; got {pretrain_dataset}."
        )

    prefix = Path(__file__).parent.joinpath("pretrained", pretrain_dataset.lower())
    checkpoint = prefix.joinpath("checkpoint.ckpt")
    state_dict = prefix.joinpath("dataset_state_dict.yaml")

    model = load_pretrained_model(checkpoint, map_location=device)
    loader = load_uspto_dataset(smiles_reactions, state_dict_path=state_dict)

    # evaluation or train mode
    if mode.lower() == "eval":
        model.eval()
        with torch.no_grad():
            rxnrep_fingerprints = get_predictions(model, loader)
    elif mode.lower() == "train":
        model.train()
        rxnrep_fingerprints = get_predictions(model, loader)
    else:
        raise ValueError(f"Expect `eval` or `train` mode; got `{mode}`")

    return rxnrep_fingerprints


def _check_species(species: List[str], supported_species: List[str]):
    not_supported = set(species) - set(supported_species)
    if not_supported:
        not_supported = ", ".join(sorted(not_supported))
        supported = ", ".join(supported_species)
        raise DatasetError(
            f"Model trained with a dataset having species: {supported}; Cannot make "
            f"predictions for molecule containing species: {not_supported}."
        )


class DatasetError(Exception):
    def __init__(self, msg=None):
        super(DatasetError, self).__init__(msg)
        self.msg = msg


if __name__ == "__main__":

    rxn1 = "[CH3:6][CH2:7][OH:16].[O:1]=[C:2]([C:3](=[O:4])[OH:5])[CH2:8][CH2:9][c:10]1[cH:11][cH:12][cH:13][cH:14][cH:15]1>>[O:1]=[C:2]([C:3](=[O:4])[O:5][CH2:6][CH3:7])[CH2:8][CH2:9][c:10]1[cH:11][cH:12][cH:13][cH:14][cH:15]1.[OH2:16]"
    rxn2 = "[C:1](#[N:2])[c:3]1[cH:4][cH:5][c:6]([CH2:7][C:8](=[O:9])[OH:10])[cH:13][cH:14]1.[CH3:11][CH2:12][OH:15]>>[C:1](#[N:2])[c:3]1[cH:4][cH:5][c:6]([CH2:7][C:8](=[O:9])[O:10][CH2:11][CH3:12])[cH:13][cH:14]1.[OH2:15]"

    reactions = [rxn1, rxn2]
    fingerprints = get_rxnrep_fingerprint(reactions)

    print(fingerprints.shape)
