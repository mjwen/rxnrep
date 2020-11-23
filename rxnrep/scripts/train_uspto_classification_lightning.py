import warnings
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl

from rxnrep.data.uspto import SchneiderDataset
from rxnrep.data.featurizer import AtomFeaturizer, BondFeaturizer, GlobalFeaturizer
from rxnrep.model.model import LinearClassification


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
    parser.add_argument(
        "--pretrained-model-filename",
        type=str,
        default=None,
        help="Path to the checkpoint of the pretrained model to use part of its "
        "parameters. If `None`, will not use it.",
    )

    parser.add_argument("--restore", type=int, default=0, help="restore training")
    parser.add_argument(
        "--dataset-state-dict-filename", type=str, default="dataset_state_dict.yaml"
    )

    # ========== distributed ==========
    parser.add_argument(
        "--launch-mode",
        type=str,
        default="torch_launch",
        help="How to launch distributed training: [`torch_launch`| `srun` | `spawn`]",
    )
    parser.add_argument("--gpu", type=int, default=0, help="Whether to use GPU.")
    parser.add_argument(
        "--distributed", type=int, default=0, help="Whether distributed DDP training."
    )
    parser.add_argument(
        "--dist-url",
        type=str,
        help="tcp port for distributed communication e.g. `tcp://localhost:15678`.",
    )
    parser.add_argument("--local_rank", type=int, help="Local rank of process.")

    args = parser.parse_args()

    return args


class LightningModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.model = LinearClassification(
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
            # classification head
            head_hidden_layer_sizes=args.head_hidden_layer_sizes,
            num_classes=args.num_classes,
            head_activation=args.head_activation,
        )

        # self.class_weights = train_loader.dataset.get_class_weight(
        #     num_classes=args.num_classes
        # )

    def forward(self, x):
        nodes = ["atom", "bond", "global"]

        indices, mol_graphs, rxn_graphs, labels, metadata = x
        feats = {nt: mol_graphs.nodes[nt].data.pop("feat") for nt in nodes}
        labels = labels["reaction_class"]

        return self.model(mol_graphs, rxn_graphs, feats, metadata)

    def training_step(self, batch, batch_idx):
        nodes = ["atom", "bond", "global"]

        indices, mol_graphs, rxn_graphs, labels, metadata = batch

        feats = {nt: mol_graphs.nodes[nt].data.pop("feat") for nt in nodes}
        labels = labels["reaction_class"]

        preds = self.model(mol_graphs, rxn_graphs, feats, metadata)

        loss = F.cross_entropy(
            #    preds, labels, reduction="mean", weight=self.class_weight
            preds,
            labels,
            reduction="mean",
        )

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        nodes = ["atom", "bond", "global"]

        indices, mol_graphs, rxn_graphs, labels, metadata = batch

        feats = {nt: mol_graphs.nodes[nt].data.pop("feat") for nt in nodes}
        labels = labels["reaction_class"]

        preds = self.model(mol_graphs, rxn_graphs, feats, metadata)

        loss = F.cross_entropy(
            #    preds, labels, reduction="mean", weight=self.class_weight
            preds,
            labels,
            reduction="mean",
        )

        self.log("val_loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        nodes = ["atom", "bond", "global"]

        indices, mol_graphs, rxn_graphs, labels, metadata = batch

        feats = {nt: mol_graphs.nodes[nt].data.pop("feat") for nt in nodes}
        labels = labels["reaction_class"]

        preds = self.model(mol_graphs, rxn_graphs, feats, metadata)

        loss = F.cross_entropy(
            #    preds, labels, reduction="mean", weight=self.class_weight
            preds,
            labels,
            reduction="mean",
        )

        self.log("test_loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

        return optimizer


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
    if not args.distributed or args.rank == 0:
        trainset.save_state_dict_file(args.dataset_state_dict_filename)
        print(
            "Trainset size: {}, valset size: {}: testset size: {}.".format(
                len(trainset), len(valset), len(testset)
            )
        )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    else:
        train_sampler = None

    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=trainset.collate_fn,
        drop_last=False,
    )

    # TODO, for val set, we can also make it distributed and report the error on rank
    #  0. If the vl set size is large enough, the statistics of error should be the same
    #  in different ranks. If this is not good, we can gather and reduce the validation
    #  metric.

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

    return train_loader, val_loader, test_loader, train_sampler


def main():
    args = parse_args()

    train_loader, val_loader, test_loader, train_sampler = load_dataset(args)

    # add args from dataset
    args.feature_size = train_loader.dataset.feature_size

    model = LightningModel(args)

    trainer = pl.Trainer(max_epochs=10, progress_bar_refresh_rate=20)
    trainer.fit(model, train_loader, val_loader)

    trainer.test(test_dataloaders=test_loader)


if __name__ == "__main__":
    main()
