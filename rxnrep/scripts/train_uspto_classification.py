import argparse
import warnings
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.dataloader import DataLoader

from rxnrep.data.featurizer import AtomFeaturizer, BondFeaturizer, GlobalFeaturizer
from rxnrep.data.uspto import SchneiderDataset
from rxnrep.model.model import LinearClassification
from rxnrep.scripts.launch_environment import PyTorchLaunch
from rxnrep.scripts.utils import (
    TimeMeter,
    get_latest_checkpoint_wandb,
    get_repo_git_commit,
    save_files_to_wandb,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Reaction Representation")

    # ========== dataset ==========
    prefix = "/Users/mjwen/Documents/Dataset/uspto/Schneider50k/"

    fname_tr = prefix + "schneider50k_n400_processed_train.tsv"
    fname_val = fname_tr
    fname_test = fname_tr

    parser.add_argument("--trainset_filename", type=str, default=fname_tr)
    parser.add_argument("--valset_filename", type=str, default=fname_val)
    parser.add_argument("--testset_filename", type=str, default=fname_test)
    parser.add_argument(
        "--dataset_state_dict_filename", type=str, default="dataset_state_dict.yaml"
    )

    # ========== model ==========
    # embedding
    parser.add_argument("--embedding_size", type=int, default=24)

    # encoder
    parser.add_argument(
        "--molecule_conv_layer_sizes", type=int, nargs="+", default=[64, 64, 64]
    )
    parser.add_argument("--molecule_num_fc_layers", type=int, default=2)
    parser.add_argument("--molecule_batch_norm", type=int, default=1)
    parser.add_argument("--molecule_activation", type=str, default="ReLU")
    parser.add_argument("--molecule_residual", type=int, default=1)
    parser.add_argument("--molecule_dropout", type=float, default="0.0")
    parser.add_argument(
        "--reaction_conv_layer_sizes", type=int, nargs="+", default=[64, 64, 64]
    )
    parser.add_argument("--reaction_num_fc_layers", type=int, default=2)
    parser.add_argument("--reaction_batch_norm", type=int, default=1)
    parser.add_argument("--reaction_activation", type=str, default="ReLU")
    parser.add_argument("--reaction_residual", type=int, default=1)
    parser.add_argument("--reaction_dropout", type=float, default="0.0")

    # linear classification head
    parser.add_argument(
        "--head_hidden_layer_sizes", type=int, nargs="+", default=[256, 128]
    )
    parser.add_argument("--head_activation", type=str, default="ReLU")
    parser.add_argument("--num_reaction_classes", type=int, default=50)

    # ========== training ==========

    # restore
    parser.add_argument("--restore", type=int, default=0, help="restore training")

    # accelerator
    parser.add_argument("--num_nodes", type=int, default=1, help="number of nodes")
    parser.add_argument(
        "--gpus", type=int, default=None, help="number of gpus per node"
    )
    parser.add_argument(
        "--accelerator", type=str, default=None, help="backend, e.g. `ddp`"
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="number of workers for dataloader"
    )
    parser.add_argument(
        "--nprocs",
        type=int,
        default=1,
        help="number of processes for constructing graphs in dataset",
    )

    # training algorithm
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=100, help="batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")

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
        num_processes=args.nprocs,
    )

    state_dict = trainset.state_dict()

    valset = SchneiderDataset(
        filename=args.valset_filename,
        atom_featurizer=AtomFeaturizer(),
        bond_featurizer=BondFeaturizer(),
        global_featurizer=GlobalFeaturizer(),
        transform_features=True,
        init_state_dict=state_dict,
        num_processes=args.nprocs,
    )

    testset = SchneiderDataset(
        filename=args.testset_filename,
        atom_featurizer=AtomFeaturizer(),
        bond_featurizer=BondFeaturizer(),
        global_featurizer=GlobalFeaturizer(),
        transform_features=True,
        init_state_dict=state_dict,
        num_processes=args.nprocs,
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
        pin_memory=True,
        num_workers=args.num_workers,
    )

    val_loader = DataLoader(
        valset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=valset.collate_fn,
        drop_last=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    test_loader = DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=testset.collate_fn,
        drop_last=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    # Add dataset state dict to args to log it
    args.dataset_state_dict = state_dict

    # Add info that will be used in the model to args for easy access
    class_weight = trainset.get_class_weight(
        num_reaction_classes=args.num_reaction_classes
    )
    args.reaction_class_weight = class_weight["reaction_class"]
    args.feature_size = trainset.feature_size

    return train_loader, val_loader, test_loader


class LightningModel(pl.LightningModule):
    def __init__(self, params):
        super().__init__()

        # save params to be accessible via self.hparams
        self.save_hyperparameters(params)
        params = self.hparams

        self.model = LinearClassification(
            in_feats=params.feature_size,
            embedding_size=params.embedding_size,
            # encoder
            molecule_conv_layer_sizes=params.molecule_conv_layer_sizes,
            molecule_num_fc_layers=params.molecule_num_fc_layers,
            molecule_batch_norm=params.molecule_batch_norm,
            molecule_activation=params.molecule_activation,
            molecule_residual=params.molecule_residual,
            molecule_dropout=params.molecule_dropout,
            reaction_conv_layer_sizes=params.reaction_conv_layer_sizes,
            reaction_num_fc_layers=params.reaction_num_fc_layers,
            reaction_batch_norm=params.reaction_batch_norm,
            reaction_activation=params.reaction_activation,
            reaction_residual=params.reaction_residual,
            reaction_dropout=params.reaction_dropout,
            # classification head
            head_hidden_layer_sizes=params.head_hidden_layer_sizes,
            num_classes=params.num_reaction_classes,
            head_activation=params.head_activation,
        )

        # metrics
        # (should be modules so that metric tensors can be placed in the correct device)
        self.metrics = nn.ModuleDict()
        for mode in ["metric_train", "metric_val", "metric_test"]:
            self.metrics[mode] = nn.ModuleDict(
                {
                    "accuracy": pl.metrics.Accuracy(compute_on_step=False),
                    "precision": pl.metrics.Precision(
                        num_classes=params.num_reaction_classes,
                        average="macro",
                        compute_on_step=False,
                    ),
                    "recall": pl.metrics.Recall(
                        num_classes=params.num_reaction_classes,
                        average="macro",
                        compute_on_step=False,
                    ),
                    "f1": pl.metrics.F1(
                        num_classes=params.num_reaction_classes,
                        average="macro",
                        compute_on_step=False,
                    ),
                }
            )

        self.timer = TimeMeter()

    def forward(self, batch):
        nodes = ["atom", "bond", "global"]

        indices, mol_graphs, rxn_graphs, labels, metadata = batch

        # lightning cannot move dgl graphs to gpu, so do it manually
        mol_graphs = mol_graphs.to(self.device)
        rxn_graphs = rxn_graphs.to(self.device)

        feats = {nt: mol_graphs.nodes[nt].data.pop("feat") for nt in nodes}

        feats, reaction_feats = self.model(mol_graphs, rxn_graphs, feats, metadata)

        return reaction_feats

    def training_step(self, batch, batch_idx):
        logits, labels, loss = self.shared_step(batch)
        self._update_metrics(logits, labels, "train")
        self.log("train/loss", loss, on_epoch=True)

        return loss

    def training_epoch_end(self, outputs):
        self._compute_metrics("train")

    def validation_step(self, batch, batch_idx):
        logits, labels, loss = self.shared_step(batch)
        self._update_metrics(logits, labels, "val")
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)

        return loss

    def validation_epoch_end(self, outputs):
        self._compute_metrics("val")

        # time it
        delta_t, cumulative_t = self.timer.update()
        self.log("epoch time", delta_t, on_step=False, on_epoch=True, prog_bar=True)
        self.log("cumulative time", cumulative_t, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        logits, labels, loss = self.shared_step(batch)
        self._update_metrics(logits, labels, "test")
        self.log("test/loss", loss, on_epoch=True)

        return loss

    def test_epoch_end(self, outputs):
        self._compute_metrics("test")

    def shared_step(self, batch):

        # ========== compute predictions ==========
        indices, mol_graphs, rxn_graphs, labels, metadata = batch

        # lightning cannot move dgl graphs to gpu, so do it manually
        mol_graphs = mol_graphs.to(self.device)
        rxn_graphs = rxn_graphs.to(self.device)

        nodes = ["atom", "bond", "global"]
        feats = {nt: mol_graphs.nodes[nt].data.pop("feat") for nt in nodes}
        labels = labels["reaction_class"]

        feats, reaction_feats = self.model(mol_graphs, rxn_graphs, feats, metadata)
        logits = self.model.decode(feats, reaction_feats)

        loss = F.cross_entropy(
            logits,
            labels,
            reduction="mean",
            weight=self.hparams.reaction_class_weight.to(self.device),
        )

        return logits, labels, loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = ReduceLROnPlateau(
            optimizer, mode="max", factor=0.4, patience=20, verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "metric_val/f1",
        }

    def _update_metrics(self, preds, labels, mode):
        """
        update metric states at each step
        """
        mode = "metric_" + mode

        for mt in self.metrics[mode]:
            metric_obj = self.metrics[mode][mt]
            metric_obj(preds, labels)

    def _compute_metrics(self, mode):
        """
        compute metric and log it at each epoch
        """
        mode = "metric_" + mode

        for mt in self.metrics[mode]:
            metric_obj = self.metrics[mode][mt]
            v = metric_obj.compute()
            self.log(f"{mode}/{mt}", v, on_step=False, on_epoch=True, prog_bar=False)

            # reset is called automatically somewhere by lightning, here we call it
            # explicitly just in case
            metric_obj.reset()


def main():
    print("\nStart training at:", datetime.now())

    pl.seed_everything(25)

    args = parse_args()

    # ========== dataset ==========
    train_loader, val_loader, test_loader = load_dataset(args)

    # ========== model ==========
    model = LightningModel(args)

    # ========== trainer ==========

    # callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="metric_val/f1", mode="max", save_last=True, save_top_k=5, verbose=False
    )
    early_stop_callback = EarlyStopping(
        monitor="metric_val/f1", min_delta=0.0, patience=50, mode="max", verbose=True
    )

    # logger
    log_save_dir = Path("wandb").resolve()
    project = "tmp-rxnrep"

    # restore model, epoch, shared_step, LR schedulers, apex, etc...
    if args.restore and log_save_dir.exists():
        # restore
        checkpoint_path = get_latest_checkpoint_wandb(log_save_dir, project)
    else:
        # create new
        checkpoint_path = None

    if not log_save_dir.exists():
        # put in try except in case it throws errors in distributed training
        try:
            log_save_dir.mkdir()
        except FileExistsError:
            pass
    wandb_logger = WandbLogger(save_dir=log_save_dir, project=project)

    # cluster environment to use torch.distributed.launch, e.g.
    # python -m torch.distributed.launch --use_env --nproc_per_node=2 <this_script.py>
    cluster = PyTorchLaunch()

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        num_nodes=args.num_nodes,
        gpus=args.gpus,
        accelerator=args.accelerator,
        plugins=[cluster],
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=wandb_logger,
        resume_from_checkpoint=checkpoint_path,
        progress_bar_refresh_rate=100,
        flush_logs_every_n_steps=50,
        weights_summary="top",
        # profiler="simple",
        # deterministic=True,
    )

    # ========== fit and test ==========
    trainer.fit(model, train_loader, val_loader)
    trainer.test(test_dataloaders=test_loader)

    # ========== save files to wandb ==========
    # Do not do this before trainer, since this might result in the initialization of
    # multiple wandb object when training in distribution mode
    if (
        args.gpus is None
        or args.gpus == 1
        or (args.gpus > 1 and cluster.local_rank() == 0)
    ):
        save_files_to_wandb(wandb_logger, __file__, ["sweep.py", "submit.sh"])

    print("\nFinish training at:", datetime.now())


if __name__ == "__main__":
    repo_path = "/Users/mjwen/Applications/rxnrep"
    latest_commit = get_repo_git_commit(repo_path)
    print(latest_commit)

    main()
