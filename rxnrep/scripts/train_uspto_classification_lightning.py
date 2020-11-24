import warnings
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl

from rxnrep.data.uspto import SchneiderDataset
from rxnrep.data.featurizer import AtomFeaturizer, BondFeaturizer, GlobalFeaturizer
from rxnrep.model.model import LinearClassification
from rxnrep.scripts.utils import get_latest_checkpoint_path


def parse_args():
    parser = argparse.ArgumentParser(description="Reaction Representation")

    # ========== input files ==========
    prefix = "/Users/mjwen/Documents/Dataset/uspto/Schneider50k/"

    fname_tr = prefix + "schneider_n200_processed_train_label_manipulated.tsv"
    fname_val = prefix + "schneider_n200_processed_val.tsv"
    fname_test = prefix + "schneider_n200_processed_test.tsv"

    parser.add_argument("--trainset-filename", type=str, default=fname_tr)
    parser.add_argument("--valset-filename", type=str, default=fname_val)
    parser.add_argument("--testset-filename", type=str, default=fname_test)

    # ========== embedding layer ==========
    parser.add_argument("--embedding-size", type=int, default=24)

    # ========== encoder ==========
    parser.add_argument(
        "--molecule-conv-layer-sizes", type=int, nargs="+", default=[64, 64, 64]
    )
    parser.add_argument("--molecule-num-fc-layers", type=int, default=2)
    parser.add_argument("--molecule-batch-norm", type=int, default=1)
    parser.add_argument("--molecule-activation", type=str, default="ReLU")
    parser.add_argument("--molecule-residual", type=int, default=1)
    parser.add_argument("--molecule-dropout", type=float, default="0.0")
    parser.add_argument(
        "--reaction-conv-layer-sizes", type=int, nargs="+", default=[64, 64, 64]
    )
    parser.add_argument("--reaction-num-fc-layers", type=int, default=2)
    parser.add_argument("--reaction-batch-norm", type=int, default=1)
    parser.add_argument("--reaction-activation", type=str, default="ReLU")
    parser.add_argument("--reaction-residual", type=int, default=1)
    parser.add_argument("--reaction-dropout", type=float, default="0.0")

    # linear classification head
    parser.add_argument(
        "--head-hidden-layer-sizes", type=int, nargs="+", default=[256, 128]
    )
    parser.add_argument("--head-activation", type=str, default="ReLU")
    parser.add_argument("--num-classes", type=int, default=50)

    parser.add_argument(
        "--pretrained-model-checkpoint",
        type=str,
        default=None,
        help="Path to the checkpoint of the pretrained model to use part of its "
        "parameters. If `None`, will not use it.",
    )
    parser.add_argument(
        "--only-train-classification-head",
        type=int,
        default=0,
        help="whether to only train the classification head",
    )

    # ========== training ==========
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--batch-size", type=int, default=100, help="batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay")

    parser.add_argument("--restore", type=int, default=0, help="restore training")
    parser.add_argument(
        "--dataset-state-dict-filename", type=str, default="dataset_state_dict.yaml"
    )

    args = parser.parse_args()

    return args


def load_dataset(args):

    # check dataset state dict if restore model
    if args.restore:
        if args.dataset_state_dict_filename is None:
            warnings.warn(
                "Restore with `args.dataset_state_dict_filename` set to None."
            )
            state_dict_filename = None
        elif not Path(args.dataset_state_dict_filename).exists():
            warnings.warn(
                f"args.dataset_state_dict_filename: `{args.dataset_state_dict_filename} "
                "not found; set to `None`."
            )
            state_dict_filename = None
        else:
            state_dict_filename = args.dataset_state_dict_filename
    else:
        state_dict_filename = None

    trainset = SchneiderDataset(
        filename=args.trainset_filename,
        atom_featurizer=AtomFeaturizer(),
        bond_featurizer=BondFeaturizer(),
        global_featurizer=GlobalFeaturizer(),
        transform_features=True,
        init_state_dict=state_dict_filename,
    )
    valset = SchneiderDataset(
        filename=args.valset_filename,
        atom_featurizer=AtomFeaturizer(),
        bond_featurizer=BondFeaturizer(),
        global_featurizer=GlobalFeaturizer(),
        transform_features=True,
        init_state_dict=trainset.state_dict(),
    )
    testset = SchneiderDataset(
        filename=args.testset_filename,
        atom_featurizer=AtomFeaturizer(),
        bond_featurizer=BondFeaturizer(),
        global_featurizer=GlobalFeaturizer(),
        transform_features=True,
        init_state_dict=trainset.state_dict(),
    )

    # save dataset state dict for retraining or prediction
    trainset.save_state_dict_file(args.dataset_state_dict_filename)
    print(
        "Trainset size: {}, valset size: {}: testset size: {}.".format(
            len(trainset), len(valset), len(testset)
        )
    )

    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=trainset.collate_fn,
        drop_last=False,
    )

    # larger val and test set batch_size is faster but needs more memory
    # adjust the batch size of to fit memory
    bs = max(len(valset) // 10, 1)
    val_loader = DataLoader(
        valset,
        batch_size=bs,
        shuffle=False,
        collate_fn=valset.collate_fn,
        drop_last=False,
    )
    bs = max(len(testset) // 10, 1)
    test_loader = DataLoader(
        testset,
        batch_size=bs,
        shuffle=False,
        collate_fn=testset.collate_fn,
        drop_last=False,
    )

    # set args for model
    args.feature_size = trainset.feature_size
    args.class_weights = train_loader.dataset.get_class_weight(
        num_classes=args.num_classes
    )

    return train_loader, val_loader, test_loader


class LightningModel(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters(params)

        self.model = LinearClassification(
            in_feats=self.hparams.feature_size,
            embedding_size=self.hparams.embedding_size,
            # encoder
            molecule_conv_layer_sizes=self.hparams.molecule_conv_layer_sizes,
            molecule_num_fc_layers=self.hparams.molecule_num_fc_layers,
            molecule_batch_norm=self.hparams.molecule_batch_norm,
            molecule_activation=self.hparams.molecule_activation,
            molecule_residual=self.hparams.molecule_residual,
            molecule_dropout=self.hparams.molecule_dropout,
            reaction_conv_layer_sizes=self.hparams.reaction_conv_layer_sizes,
            reaction_num_fc_layers=self.hparams.reaction_num_fc_layers,
            reaction_batch_norm=self.hparams.reaction_batch_norm,
            reaction_activation=self.hparams.reaction_activation,
            reaction_residual=self.hparams.reaction_residual,
            reaction_dropout=self.hparams.reaction_dropout,
            # classification head
            head_hidden_layer_sizes=self.hparams.head_hidden_layer_sizes,
            num_classes=self.hparams.num_classes,
            head_activation=self.hparams.head_activation,
        )

    def forward(self, x):
        nodes = ["atom", "bond", "global"]

        indices, mol_graphs, rxn_graphs, labels, metadata = x
        feats = {nt: mol_graphs.nodes[nt].data.pop("feat") for nt in nodes}
        labels = labels["reaction_class"]

        return self.model(mol_graphs, rxn_graphs, feats, metadata)

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log("val_loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log("test_loss", loss)

        return loss

    def _shared_step(self, batch, batch_idx):
        nodes = ["atom", "bond", "global"]

        indices, mol_graphs, rxn_graphs, labels, metadata = batch

        feats = {nt: mol_graphs.nodes[nt].data.pop("feat") for nt in nodes}
        labels = labels["reaction_class"]

        preds = self.model(mol_graphs, rxn_graphs, feats, metadata)

        loss = F.cross_entropy(
            preds, labels, reduction="mean", weight=self.hparams.class_weights
        )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        return optimizer


def load_pretrained_model(model, pretrained_model_checkpoint: Path, map_location=None):
    """
    Load pretrained representational learning model for fine tuning like learning
    classification head. So, this is typically used together with
    `freeze_params_other_than_classification_head(model):
    """
    checkpoints = torch.load(pretrained_model_checkpoint, map_location=None)

    # state_dict is the pretrained model's parameters, see
    # https://github.com/PyTorchLightning/pytorch-lightning/blob/42e59c6add29a5f91654a9c3a76febbe435df981/pytorch_lightning/trainer/connectors/checkpoint_connector.py#L255
    pretrained_dict = checkpoints["state_dict"]
    model_dict = model.state_dict()

    # filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)

    # load the new state dict
    model.load_state_dict(model_dict)
    print("\nLoad pretrained model...")


def freeze_params_other_than_classification_head(model):
    """
    Free encoder and set2set parameters so that we train classification head only.
    """

    # freeze encoder parameters
    for p in model.model.encoder.parameters():
        p.requires_grad = False
    # freeze set2set parameters
    for p in model.model.set2set.parameters():
        p.requires_grad = False

    # check only classification head params is trained
    num_params_classification_head = sum(
        p.numel() for p in model.model.classification_head.parameters()
    )
    num_params_trainable = sum(
        [p.numel() for p in model.model.parameters() if p.requires_grad]
    )
    assert (
        num_params_classification_head == num_params_trainable
    ), "parameters other than classification head are trained"


def main():

    print("\nStart training at:", datetime.now())

    pl.seed_everything(25)

    # ========== args ==========
    args = parse_args()

    # ========== dataset ==========
    train_loader, val_loader, test_loader = load_dataset(args)

    # ========== model ==========
    model = LightningModel(args)

    # load pretrained models
    if args.pretrained_model_checkpoint is not None:
        load_pretrained_model(model, args.pre)
        print("\nLoad pretrained model...")

    # freeze parameters
    if args.only_train_classification_head:
        freeze_params_other_than_classification_head(model)
        print("\nFreeze some parameters to only train classification head...")

    # ========== trainer ==========

    # restores model, epoch, step, LR schedulers, apex, etc...
    if args.restore and Path("./lightning_logs").resolve().exists():
        checkpoint_path = get_latest_checkpoint_path("./lightning_logs")
    # create new
    else:
        checkpoint_path = None
    trainer = pl.Trainer(
        max_epochs=10,
        resume_from_checkpoint=checkpoint_path,
        progress_bar_refresh_rate=10,
    )

    # ========== fit and test ==========
    trainer.fit(model, train_loader, val_loader)
    trainer.test(test_dataloaders=test_loader)

    print("\nFinish training at:", datetime.now())


if __name__ == "__main__":
    main()
