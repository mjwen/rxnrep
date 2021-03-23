import argparse
from datetime import datetime

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from rxnrep.model.clustering import DistributedReactionCluster, ReactionCluster
from rxnrep.model.model import ReactionRepresentation
from rxnrep.scripts.load_dataset import load_uspto_dataset
from rxnrep.scripts.main import main
from rxnrep.scripts.utils import TimeMeter, write_running_metadata


class RxnRepLightningModel(pl.LightningModule):
    def __init__(self, params):
        super().__init__()

        # save params to be accessible via self.hparams
        self.save_hyperparameters(params)
        params = self.hparams

        self.model = ReactionRepresentation(
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
            # bond hop distance decoder
            bond_hop_dist_decoder_hidden_layer_sizes=params.node_decoder_hidden_layer_sizes,
            bond_hop_dist_decoder_activation=params.node_decoder_activation,
            bond_hop_dist_decoder_num_classes=params.bond_hop_dist_num_classes,
            # atom hop distance decoder
            atom_hop_dist_decoder_hidden_layer_sizes=params.node_decoder_hidden_layer_sizes,
            atom_hop_dist_decoder_activation=params.node_decoder_activation,
            atom_hop_dist_decoder_num_classes=params.atom_hop_dist_num_classes,
            # masked atom type decoder
            masked_atom_type_decoder_hidden_layer_sizes=params.node_decoder_hidden_layer_sizes,
            masked_atom_type_decoder_activation=params.node_decoder_activation,
            masked_atom_type_decoder_num_classes=params.masked_atom_type_num_classes,
            # pool method
            pool_method=params.pool_method,
            pool_kwargs=params.pool_kwargs,
        )

        # reaction cluster functions
        self.reaction_cluster_fn = {mode: None for mode in ["train", "val", "test"]}
        self.assignments = {mode: None for mode in ["train", "val", "test"]}
        self.centroids = None

        # metrics
        self.metrics = self._init_metrics()

        self.timer = TimeMeter()

    def forward(self, batch, returns: str = "reaction_feature"):
        """
        Args:
            batch:
            returns: the type of features (embeddings) to return. Optionals are
                `reaction_feature`, 'diff_feature_before_rxn_conv',
                and 'diff_feature_after_rxn_conv'.

        Returns:
            If returns = `reaction_feature`, return a 2D tensor of reaction features,
            each row for a reaction;
            If returns = `diff_feature_before_rxn_conv` or `diff_feature_after_rxn_conv`,
                return a dictionary of atom, bond, and global features.
                As the name suggests, the returned features can be `before` or `after`
                the reaction conv layers.
        """
        nodes = ["atom", "bond", "global"]

        indices, mol_graphs, rxn_graphs, labels, metadata = batch

        # lightning cannot move dgl graphs to gpu, so do it manually
        mol_graphs = mol_graphs.to(self.device)
        rxn_graphs = rxn_graphs.to(self.device)

        feats = {nt: mol_graphs.nodes[nt].data.pop("feat") for nt in nodes}

        if returns == "reaction_feature":
            _, reaction_feats = self.model(mol_graphs, rxn_graphs, feats, metadata)
            return reaction_feats

        elif returns == "diff_feature_after_rxn_conv":
            diff_feats, _ = self.model(mol_graphs, rxn_graphs, feats, metadata)
            return diff_feats

        elif returns == "diff_feature_before_rxn_conv":
            diff_feats = self.model.get_diff_feats(
                mol_graphs, rxn_graphs, feats, metadata
            )
            return diff_feats

        else:
            supported = [
                "reaction_feature",
                "diff_feature_before_rxn_conv",
                "diff_feature_after_rxn_conv",
            ]
            raise ValueError(
                f"Expect `returns` to be one of {supported}; got `{returns}`."
            )

    def on_train_epoch_start(self):
        if self.reaction_cluster_fn["train"] is None:
            self.reaction_cluster_fn["train"] = self._init_reaction_cluster_fn(
                self.train_dataloader()
            )

        assi, cent = self.reaction_cluster_fn["train"].get_cluster_assignments(
            centroids="random", predict_only=False
        )
        self.assignments["train"] = assi
        self.centroids = cent

    def training_step(self, batch, batch_idx):
        loss, preds, labels, indices = self.shared_step(batch, "train")
        self._update_metrics(preds, labels, "train")

        return {
            "loss": loss,
            "indices": indices.cpu(),
            "reaction_cluster_feats": preds["reaction_cluster"].detach().cpu(),
        }

    def training_epoch_end(self, outputs):
        self._compute_metrics("train")
        self._track_reaction_cluster_data(outputs, "train")

    def on_validation_epoch_start(self):
        if self.reaction_cluster_fn["val"] is None:
            self.reaction_cluster_fn["val"] = self._init_reaction_cluster_fn(
                self.val_dataloader()
            )

        assi, _ = self.reaction_cluster_fn["val"].get_cluster_assignments(
            centroids=self.centroids, predict_only=True
        )
        self.assignments["val"] = assi

    def validation_step(self, batch, batch_idx):
        loss, preds, labels, indices = self.shared_step(batch, "val")
        self._update_metrics(preds, labels, "val")

        return {
            "loss": loss,
            "indices": indices.cpu(),
            "reaction_cluster_feats": preds["reaction_cluster"].detach().cpu(),
        }

    def validation_epoch_end(self, outputs):
        # sum f1 to look for early stop and learning rate scheduler
        sum_f1 = self._compute_metrics("val")
        self._track_reaction_cluster_data(outputs, "val")

        self.log(f"val/f1", sum_f1, on_step=False, on_epoch=True, prog_bar=True)

        # time it
        delta_t, cumulative_t = self.timer.update()
        self.log("epoch time", delta_t, on_step=False, on_epoch=True, prog_bar=True)
        self.log("cumulative time", cumulative_t, on_step=False, on_epoch=True)

    def on_test_epoch_start(self):
        if self.reaction_cluster_fn["test"] is None:
            self.reaction_cluster_fn["test"] = self._init_reaction_cluster_fn(
                self.test_dataloader()
            )

        assi, _ = self.reaction_cluster_fn["test"].get_cluster_assignments(
            centroids=self.centroids, predict_only=True
        )
        self.assignments["test"] = assi

    def test_step(self, batch, batch_idx):
        loss, preds, labels, indices = self.shared_step(batch, "test")
        self._update_metrics(preds, labels, "test")

        return {
            "loss": loss,
            "indices": indices.cpu(),
            "reaction_cluster_feats": preds["reaction_cluster"].detach().cpu(),
        }

    def test_epoch_end(self, outputs):
        self._compute_metrics("test")
        self._track_reaction_cluster_data(outputs, "test")

    def shared_step(self, batch, mode):

        # ========== compute predictions ==========
        indices, mol_graphs, rxn_graphs, labels, metadata = batch

        # lightning cannot move dgl graphs to gpu, so do it manually
        mol_graphs = mol_graphs.to(self.device)
        rxn_graphs = rxn_graphs.to(self.device)

        nodes = ["atom", "bond", "global"]
        feats = {nt: mol_graphs.nodes[nt].data.pop("feat") for nt in nodes}

        feats, reaction_feats = self.model(mol_graphs, rxn_graphs, feats, metadata)
        preds = self.model.decode(feats, reaction_feats, metadata)

        # ========== compute losses ==========
        # loss for bond hop distance prediction
        loss_atom_hop = F.cross_entropy(
            preds["bond_hop_dist"],
            labels["bond_hop_dist"],
            reduction="mean",
            weight=self.hparams.bond_hop_dist_class_weight.to(self.device),
        )

        # loss for atom hop distance prediction
        loss_bond_hop = F.cross_entropy(
            preds["atom_hop_dist"],
            labels["atom_hop_dist"],
            reduction="mean",
            weight=self.hparams.atom_hop_dist_class_weight.to(self.device),
        )

        # masked atom type decoder
        loss_masked_atom_type = F.cross_entropy(
            preds["masked_atom_type"], labels["masked_atom_type"], reduction="mean"
        )

        # loss for clustering prediction
        loss_reaction_cluster = []
        for a, c in zip(self.assignments[mode], self.centroids):
            a = a[indices].to(self.device)  # select for current batch from all
            c = c.to(self.device)
            p = torch.mm(preds["reaction_cluster"], c.t()) / self.hparams.temperature
            e = F.cross_entropy(p, a)
            loss_reaction_cluster.append(e)
        loss_reaction_cluster = sum(loss_reaction_cluster) / len(loss_reaction_cluster)

        # total loss (maybe assign different weights)
        loss = (
            loss_atom_hop
            + loss_bond_hop
            + loss_masked_atom_type
            + loss_reaction_cluster
        )

        # ========== log the loss ==========
        self.log_dict(
            {
                f"{mode}/loss/bond_hop_dist": loss_atom_hop,
                f"{mode}/loss/atom_hop_dist": loss_bond_hop,
                f"{mode}/loss/masked_atom_type": loss_masked_atom_type,
                f"{mode}/loss/reaction_cluster": loss_reaction_cluster,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(f"{mode}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss, preds, labels, indices

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

    def _init_reaction_cluster_fn(self, dataloader):
        # distributed
        if self.use_ddp:
            reaction_cluster_fn = DistributedReactionCluster(
                self.model,
                dataloader,
                num_centroids=self.hparams.num_centroids,
                device=self.device,
            )

        # single process
        else:
            reaction_cluster_fn = ReactionCluster(
                self.model,
                dataloader,
                num_centroids=self.hparams.num_centroids,
                device=self.device,
            )

        return reaction_cluster_fn

    def _track_reaction_cluster_data(self, outputs, mode):
        """
        Keep track of reaction clustering data to be used in the next iteration, including
        feats used for clustering (after projection head mapping) and their indices.
        """
        indices = torch.cat([x["indices"] for x in outputs])
        feats = torch.cat([x["reaction_cluster_feats"] for x in outputs])
        self.reaction_cluster_fn[mode].set_local_data_and_index(feats, indices)

    def _init_metrics(self):
        # metrics should be modules so that metric tensors can be placed in the correct
        # device

        metrics = nn.ModuleDict()
        for mode in ["metric_train", "metric_val", "metric_test"]:
            metrics[mode] = nn.ModuleDict(
                {
                    "bond_hop_dist": nn.ModuleDict(
                        {
                            "accuracy": pl.metrics.Accuracy(compute_on_step=False),
                            "precision": pl.metrics.Precision(
                                num_classes=self.hparams.bond_hop_dist_num_classes,
                                average="macro",
                                compute_on_step=False,
                            ),
                            "recall": pl.metrics.Recall(
                                num_classes=self.hparams.bond_hop_dist_num_classes,
                                average="macro",
                                compute_on_step=False,
                            ),
                            "f1": pl.metrics.F1(
                                num_classes=self.hparams.bond_hop_dist_num_classes,
                                average="macro",
                                compute_on_step=False,
                            ),
                        }
                    ),
                    "atom_hop_dist": nn.ModuleDict(
                        {
                            "accuracy": pl.metrics.Accuracy(compute_on_step=False),
                            "precision": pl.metrics.Precision(
                                num_classes=self.hparams.atom_hop_dist_num_classes,
                                average="macro",
                                compute_on_step=False,
                            ),
                            "recall": pl.metrics.Recall(
                                num_classes=self.hparams.atom_hop_dist_num_classes,
                                average="macro",
                                compute_on_step=False,
                            ),
                            "f1": pl.metrics.F1(
                                num_classes=self.hparams.atom_hop_dist_num_classes,
                                average="macro",
                                compute_on_step=False,
                            ),
                        }
                    ),
                    "masked_atom_type": nn.ModuleDict(
                        {
                            "accuracy": pl.metrics.Accuracy(compute_on_step=False),
                            "precision": pl.metrics.Precision(
                                num_classes=self.hparams.masked_atom_type_num_classes,
                                average="macro",
                                compute_on_step=False,
                            ),
                            "recall": pl.metrics.Recall(
                                num_classes=self.hparams.masked_atom_type_num_classes,
                                average="macro",
                                compute_on_step=False,
                            ),
                            "f1": pl.metrics.F1(
                                num_classes=self.hparams.masked_atom_type_num_classes,
                                average="macro",
                                compute_on_step=False,
                            ),
                        }
                    ),
                }
            )

        return metrics

    def _update_metrics(
        self,
        preds,
        labels,
        mode,
        keys=("bond_hop_dist", "atom_hop_dist", "masked_atom_type"),
    ):
        """
        update metric states at each step
        """
        mode = "metric_" + mode

        for key in keys:
            for mt in self.metrics[mode][key]:
                metric_obj = self.metrics[mode][key][mt]
                metric_obj(preds[key], labels[key])

    def _compute_metrics(
        self,
        mode,
        keys=("bond_hop_dist", "atom_hop_dist", "masked_atom_type"),
    ):
        """
        compute metric and log it at each epoch
        """
        mode = "metric_" + mode

        sum_f1 = 0
        for key in keys:
            for name in self.metrics[mode][key]:

                metric_obj = self.metrics[mode][key][name]
                value = metric_obj.compute()

                self.log(
                    f"{mode}/{name}/{key}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )

                # reset is called automatically somewhere by lightning, here we call it
                # explicitly just in case
                metric_obj.reset()

                if name == "f1":
                    sum_f1 += value

        return sum_f1


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
    parser.add_argument("--molecule_dropout", type=float, default="0.0")
    parser.add_argument(
        "--reaction_conv_layer_sizes", type=int, nargs="+", default=[64, 64, 64]
    )
    parser.add_argument("--reaction_num_fc_layers", type=int, default=2)
    parser.add_argument("--reaction_batch_norm", type=int, default=1)
    parser.add_argument("--reaction_activation", type=str, default="ReLU")
    parser.add_argument("--reaction_residual", type=int, default=1)
    parser.add_argument("--reaction_dropout", type=float, default="0.0")

    # ========== pool ==========
    parser.add_argument(
        "--pool_method",
        type=str,
        default="set2set",
        help="set2set or hop_distance",
    )

    # ========== decoder ==========
    # atom and bond decoder
    parser.add_argument(
        "--node_decoder_hidden_layer_sizes", type=int, nargs="+", default=[64]
    )
    parser.add_argument("--node_decoder_activation", type=str, default="ReLU")
    parser.add_argument("--max_hop_distance", type=int, default=3)
    parser.add_argument("--atom_type_masker_ratio", type=float, default=0.2)
    parser.add_argument(
        "--atom_type_masker_use_masker_value",
        type=int,
        default=1,
        help="whether to use atom type masker value",
    )

    # clustering decoder
    parser.add_argument(
        "--cluster_decoder_hidden_layer_sizes", type=int, nargs="+", default=[64]
    )
    parser.add_argument("--cluster_decoder_activation", type=str, default="ReLU")
    parser.add_argument(
        "--cluster_decoder_projection_head_size",
        type=int,
        default=30,
        help="projection head size for the clustering decoder",
    )
    parser.add_argument(
        "--num_centroids",
        type=int,
        nargs="+",
        default=[10],
        help="number of centroids for each clustering prototype",
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

    ####################
    # helper args
    ####################
    # encoder
    parser.add_argument(
        "--conv_layer_size",
        type=int,
        default=64,
        help="hidden layer size for mol and rxn conv",
    )
    parser.add_argument("--num_mol_conv_layers", type=int, default=2)
    parser.add_argument("--num_rxn_conv_layers", type=int, default=2)

    # node decoder
    parser.add_argument("--num_node_decoder_layers", type=int, default=1)

    # cluster decoder
    parser.add_argument("--num_cluster_decoder_layers", type=int, default=1)
    parser.add_argument("--prototype_size", type=int, default=10)
    parser.add_argument("--num_prototypes", type=int, default=1)

    ####################
    args = parser.parse_args()
    ####################

    ####################
    # adjust args
    ####################
    # encoder
    args.molecule_conv_layer_sizes = [args.conv_layer_size] * args.num_mol_conv_layers
    args.reaction_conv_layer_sizes = [args.conv_layer_size] * args.num_rxn_conv_layers
    if args.num_rxn_conv_layers == 0:
        args.reaction_dropout = 0

    decoder_layer_size = 2 * args.conv_layer_size

    # node decoder
    args.node_decoder_hidden_layer_sizes = [
        max(decoder_layer_size // 2 ** i, 50)
        for i in range(args.num_node_decoder_layers)
    ]

    # cluster decoder
    args.cluster_decoder_hidden_layer_sizes = [
        max(decoder_layer_size // 2 ** i, 50)
        for i in range(args.num_cluster_decoder_layers)
    ]
    args.num_centroids = [args.prototype_size] * args.num_prototypes

    # adjust for pool
    if args.pool_method == "set2set":
        args.pool_kwargs = None
    elif args.pool_method == "hop_distance":
        args.pool_kwargs = {"max_hop_distance": args.max_hop_distance}

    return args


if __name__ == "__main__":

    print("Start training at:", datetime.now())

    filename = "running_metadata.yaml"
    repo_path = "/Users/mjwen/Applications/rxnrep"
    write_running_metadata(filename, repo_path)

    pl.seed_everything(25)

    # args
    args = parse_args()

    # dataset
    train_loader, val_loader, test_loader = load_uspto_dataset(args)

    # model
    model = RxnRepLightningModel(args)

    project = "tmp-rxnrep"
    main(args, model, train_loader, val_loader, test_loader, project)

    print("Finish training at:", datetime.now())
