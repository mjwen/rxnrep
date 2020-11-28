import warnings
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from rxnrep.data.uspto import USPTODataset
from rxnrep.data.featurizer import AtomFeaturizer, BondFeaturizer, GlobalFeaturizer
from rxnrep.model.model import ReactionRepresentation
from rxnrep.model.clustering import ReactionCluster, DistributedReactionCluster
from rxnrep.scripts.utils import get_latest_checkpoint_wandb
from rxnrep.scripts.utils import TimeMeter


def parse_args():
    parser = argparse.ArgumentParser(description="Reaction Representation")

    # ========== dataset ==========
    prefix = "/Users/mjwen/Documents/Dataset/uspto/raw/"

    fname_tr = prefix + "2001_Sep2016_USPTOapplications_smiles_n200_processed_train.tsv"
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
    parser.add_argument("--molecule-dropout", type=float, default="0.0")
    parser.add_argument(
        "--reaction_conv_layer_sizes", type=int, nargs="+", default=[64, 64, 64]
    )
    parser.add_argument("--reaction_num_fc_layers", type=int, default=2)
    parser.add_argument("--reaction_batch_norm", type=int, default=1)
    parser.add_argument("--reaction_activation", type=str, default="ReLU")
    parser.add_argument("--reaction_residual", type=int, default=1)
    parser.add_argument("--reaction_dropout", type=float, default="0.0")

    # ========== decoder ==========
    # atom and bond decoder
    parser.add_argument(
        "--node_decoder_hidden_layer_sizes", type=int, nargs="+", default=[64]
    )
    parser.add_argument("--node_decoder_activation", type=str, default="ReLU")

    # clustering decoder
    parser.add_argument(
        "--cluster_decoder_hidden_layer_sizes", type=int, nargs="+", default=[64]
    )
    parser.add_argument("--cluster_decoder_activation", type=str, default="ReLU")
    parser.add_argument(
        "--cluster_decoder_projection_head_size",
        type=int,
        default=33,
        help="projection head size for the clustering decoder",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature in the loss for cluster decoder",
    )

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

    trainset = USPTODataset(
        filename=args.trainset_filename,
        atom_featurizer=AtomFeaturizer(),
        bond_featurizer=BondFeaturizer(),
        global_featurizer=GlobalFeaturizer(),
        transform_features=True,
        init_state_dict=state_dict_filename,
    )

    state_dict = trainset.state_dict()

    valset = USPTODataset(
        filename=args.valset_filename,
        atom_featurizer=AtomFeaturizer(),
        bond_featurizer=BondFeaturizer(),
        global_featurizer=GlobalFeaturizer(),
        transform_features=True,
        init_state_dict=state_dict,
    )

    testset = USPTODataset(
        filename=args.testset_filename,
        atom_featurizer=AtomFeaturizer(),
        bond_featurizer=BondFeaturizer(),
        global_featurizer=GlobalFeaturizer(),
        transform_features=True,
        init_state_dict=state_dict,
    )

    # TODO should be done by only rank 0, maybe move to prepare_data() of model
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
    )

    val_loader = DataLoader(
        valset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=valset.collate_fn,
        drop_last=False,
        pin_memory=True,
    )
    test_loader = DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=testset.collate_fn,
        drop_last=False,
        pin_memory=True,
    )

    # Add dataset state dict to args to log it
    args.dataset_state_dict = state_dict

    # Add info that will be used in the model to args for easy access
    args.feature_size = trainset.feature_size
    args.bond_type_decoder_num_classes = 3
    (
        args.atom_in_reaction_center_class_weight,
        args.bond_type_class_weight,
    ) = trainset.get_atom_in_reaction_center_and_bond_type_class_weight()

    return train_loader, val_loader, test_loader


class LightningModel(pl.LightningModule):
    def __init__(self, params):
        super().__init__()

        # save params to be accessible via self.hparams
        self.save_hyperparameters(params)

        self.model = ReactionRepresentation(
            in_feats=self.hparams.feature_size,
            embedding_size=self.hparams.embedding_size,
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
            # bond type decoder
            bond_type_decoder_hidden_layer_sizes=params.node_decoder_hidden_layer_sizes,
            bond_type_decoder_activation=params.node_decoder_activation,
            # atom in reaction center decoder
            atom_in_reaction_center_decoder_hidden_layer_sizes=params.node_decoder_hidden_layer_sizes,
            atom_in_reaction_center_decoder_activation=params.node_decoder_activation,
            # clustering decoder
            reaction_cluster_decoder_hidden_layer_sizes=params.cluster_decoder_hidden_layer_sizes,
            reaction_cluster_decoder_activation=params.cluster_decoder_activation,
            reaction_cluster_decoder_output_size=params.cluster_decoder_projection_head_size,
            # bond type decoder
            bond_type_decoder_num_classes=params.bond_type_decoder_num_classes,
        )

        # reaction cluster
        self.train_reaction_cluster = None
        self.val_reaction_cluster = None
        self.test_reaction_cluster = None

        # metrics
        for i in range(3):
            m = {
                "bond_type": {
                    "accuracy": pl.metrics.Accuracy(compute_on_step=False),
                    "precision": pl.metrics.Precision(
                        num_classes=3, compute_on_step=False
                    ),
                    "recall": pl.metrics.Recall(num_classes=3, compute_on_step=False),
                    "f1": pl.metrics.F1(num_classes=3, compute_on_step=False),
                },
                # binary classification, so num_classes = 1
                "atom_in_reaction_center": {
                    "accuracy": pl.metrics.Accuracy(
                        threshold=0.5, compute_on_step=False
                    ),
                    "precision": pl.metrics.Precision(
                        num_classes=1, threshold=0.5, compute_on_step=False
                    ),
                    "recall": pl.metrics.Recall(
                        num_classes=1, threshold=0.5, compute_on_step=False
                    ),
                    "f1": pl.metrics.F1(
                        num_classes=1, threshold=0.5, compute_on_step=False
                    ),
                },
            }
            if i == 0:
                self.train_metrics = m
            elif i == 1:
                self.val_metrics = m
            else:
                self.test_metrics = m

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

    def on_fit_start(self):
        if self.train_reaction_cluster is None:
            self.train_reaction_cluster = self._init_reaction_cluster(
                self.train_dataloader()
            )

        if self.val_reaction_cluster is None:
            self.val_reaction_cluster = self._init_reaction_cluster(
                self.val_dataloader()
            )

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        # cluster to get assignments and centroids

        (
            self.train_assignments,
            self.train_centroids,
        ) = self.train_reaction_cluster.get_cluster_assignments()

    def training_step(self, batch, batch_idx):
        indices, preds, labels, loss = self.shared_step(
            batch, self.train_assignments, self.train_centroids
        )

        # update metric states
        keys = ["bond_type", "atom_in_reaction_center"]
        for key in keys:
            for metric in self.train_metrics[key]:
                self.train_metrics[key][metric].update(preds[key], labels[key])

        # set on_epoch=True, such that the loss of each step is aggregated (average by
        # default) and logged at each epoch
        self.log("train/loss", loss, on_epoch=True)

        return {
            "loss": loss,
            "indices": indices.cpu(),
            "data_for_cluster": preds["reaction_cluster"].detach().cpu(),
        }

    def training_epoch_end(self, outputs):
        # compute metric (using all data points)
        keys = ["bond_type", "atom_in_reaction_center"]
        for key in keys:
            for metric in self.train_metrics[key]:
                v = self.train_metrics[key][metric].compute()
                self.log(f"train/{key}_{metric}", v, prog_bar=True)

        # keep track of clustering data to be used in the next iteration
        indices = torch.cat([x["indices"] for x in outputs])
        data_for_cluster = torch.cat([x["data_for_cluster"] for x in outputs])
        self.train_reaction_cluster.set_local_data_and_index(data_for_cluster, indices)

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        # cluster to get assignments and centroids

        (
            self.val_assignments,
            self.val_centroids,
        ) = self.val_reaction_cluster.get_cluster_assignments()

    def validation_step(self, batch, batch_idx):
        indices, preds, labels, loss = self.shared_step(
            batch, self.val_assignments, self.val_centroids
        )

        # update metric states
        keys = ["bond_type", "atom_in_reaction_center"]
        for key in keys:
            for metric in self.val_metrics[key]:
                self.val_metrics[key][metric].update(preds[key], labels[key])

        self.log("val/loss", loss, on_epoch=True, prog_bar=True)

        return {
            "loss": loss,
            "indices": indices.cpu(),
            "data_for_cluster": preds["reaction_cluster"].detach().cpu(),
        }

    def validation_epoch_end(self, outputs):
        # compute metric (using all data points)

        # total f1 to look for early stop and learning rate scheduler
        val_f1 = 0

        keys = ["bond_type", "atom_in_reaction_center"]
        for key in keys:
            for metric in self.val_metrics[key]:
                v = self.val_metrics[key][metric].compute()
                self.log(f"val/{key}_{metric}", v, prog_bar=True)
                if metric == "f1":
                    val_f1 += v

        # keep track of clustering data to be used in the next iteration
        indices = torch.cat([x["indices"] for x in outputs])
        data_for_cluster = torch.cat([x["data_for_cluster"] for x in outputs])
        self.val_reaction_cluster.set_local_data_and_index(data_for_cluster, indices)

        self.log(f"val/f1", val_f1, prog_bar=True)

        # time it
        delta_t, cumulative_t = self.timer.update()
        self.log("epoch time", delta_t)
        self.log("cumulative time", cumulative_t)

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx):
        # cluster to get assignments and centroids
        if self.test_reaction_cluster is None:
            self.test_reaction_cluster = self._init_reaction_cluster(
                self.test_dataloader()
            )
        (
            self.test_assignments,
            self.test_centroids,
        ) = self.test_reaction_cluster.get_cluster_assignments()

    def test_step(self, batch, batch_idx):
        indices, preds, labels, loss = self.shared_step(
            batch, self.train_assignments, self.train_centroids
        )

        # update metric states
        keys = ["bond_type", "atom_in_reaction_center"]
        for key in keys:
            for metric in self.test_metrics[key]:
                self.test_metrics[key][metric].update(preds[key], labels[key])

        self.log("test/loss", loss, on_epoch=True, prog_bar=True)

        return {
            "loss": loss,
            "indices": indices.cpu(),
            "data_for_cluster": preds["reaction_cluster"].detach().cpu(),
        }

    def test_epoch_end(self, outputs):
        # compute metric (using all data points)
        keys = ["bond_type", "atom_in_reaction_center"]
        for key in keys:
            for metric in self.test_metrics[key]:
                v = self.test_metrics[key][metric].compute()
                self.log(f"test/{key}_{metric}", v, prog_bar=True)

        # keep track of clustering data to be used in the next iteration
        indices = torch.cat([x["indices"] for x in outputs])
        data_for_cluster = torch.cat([x["data_for_cluster"] for x in outputs])
        self.test_reaction_cluster.set_local_data_and_index(data_for_cluster, indices)

    def _init_reaction_cluster(self, dataloader):

        # TODO, reaction cluster requires data loader now, so we init it here.
        #  need to move it to __init__ once we remove dataloader from the initializer
        # ALSO, make num_centroids a hyperparams

        # distributed
        if self.use_ddp:
            reaction_cluster = DistributedReactionCluster(
                self.model,
                dataloader,
                self.hparams.batch_size,
                num_centroids=[22, 22],
                device=self.device,
            )
        # single process
        else:
            reaction_cluster = ReactionCluster(
                self.model,
                dataloader.dataset,
                self.hparams.batch_size,
                num_centroids=[22, 22],
                device=self.device,
            )

        return reaction_cluster

    def shared_step(self, batch, cluster_assignments, cluster_centroids):
        nodes = ["atom", "bond", "global"]

        indices, mol_graphs, rxn_graphs, labels, metadata = batch

        # lightning cannot move dgl graphs to gpu, so do it manually
        mol_graphs = mol_graphs.to(self.device)
        rxn_graphs = rxn_graphs.to(self.device)

        feats = {nt: mol_graphs.nodes[nt].data.pop("feat") for nt in nodes}

        feats, reaction_feats = self.model(mol_graphs, rxn_graphs, feats, metadata)
        preds = self.model.decode(feats, reaction_feats)

        # ========== loss for bond type prediction ==========
        loss_bond_type = F.cross_entropy(
            preds["bond_type"],
            labels["bond_type"],
            reduction="mean",
            weight=torch.as_tensor(
                self.hparams.bond_type_class_weight, device=self.device
            ),
        )

        # ========== loss for atom in reaction center prediction ==========
        preds["atom_in_reaction_center"] = preds["atom_in_reaction_center"].flatten()
        loss_atom_in_reaction_center = F.binary_cross_entropy_with_logits(
            preds["atom_in_reaction_center"],
            labels["atom_in_reaction_center"],
            reduction="mean",
            pos_weight=torch.as_tensor(
                self.hparams.atom_in_reaction_center_class_weight[0], device=self.device
            ),
        )

        # ========== loss for clustering prediction ==========
        loss_reaction_cluster = []
        for a, c in zip(cluster_assignments, cluster_centroids):
            a = a[indices].to(self.device)  # select for current batch from all
            c = c.to(self.device)
            p = torch.mm(preds["reaction_cluster"], c.t()) / self.hparams.temperature
            e = F.cross_entropy(p, a)
            loss_reaction_cluster.append(e)
        loss_reaction_cluster = sum(loss_reaction_cluster) / len(loss_reaction_cluster)

        # TODO maybe assign different weights
        # total loss
        loss = loss_bond_type + loss_atom_in_reaction_center + loss_reaction_cluster

        return indices, preds, labels, loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = ReduceLROnPlateau(
            optimizer, mode="max", factor=0.4, patience=20, verbose=True
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val/f1"}


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
        monitor="val/f1", mode="max", save_last=True, save_top_k=5, verbose=False
    )
    early_stop_callback = EarlyStopping(
        monitor="val/f1", min_delta=0.0, patience=50, mode="min", verbose=True
    )

    # logger
    log_save_dir = Path("wandb").resolve()
    project = "schneider-classification"

    # restore model, epoch, shared_step, LR schedulers, apex, etc...
    if args.restore and log_save_dir.exists():
        # restore
        checkpoint_path = get_latest_checkpoint_wandb(log_save_dir, project)
    else:
        # create new
        checkpoint_path = None

    if not log_save_dir.exists():
        log_save_dir.mkdir()
    # wandb_logger = WandbLogger(save_dir=log_save_dir, project=project)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        num_nodes=args.num_nodes,
        gpus=args.gpus,
        accelerator=args.accelerator,
        progress_bar_refresh_rate=5,
        resume_from_checkpoint=checkpoint_path,
        callbacks=[checkpoint_callback, early_stop_callback],
        #    logger=wandb_logger,
        flush_logs_every_n_steps=50,
        weights_summary="top",
        # profiler="simple",
        # deterministic=True,
    )

    # ========== fit and test ==========
    trainer.fit(model, train_loader, val_loader)
    trainer.test(test_dataloaders=test_loader)

    print("\nFinish training at:", datetime.now())


if __name__ == "__main__":
    main()
