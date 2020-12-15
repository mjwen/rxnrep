import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from rxnrep.model.clustering import DistributedReactionCluster, ReactionCluster
from rxnrep.model.model import ReactionRepresentation
from rxnrep.scripts.utils import TimeMeter


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
            # clustering decoder
            reaction_cluster_decoder_hidden_layer_sizes=params.cluster_decoder_hidden_layer_sizes,
            reaction_cluster_decoder_activation=params.cluster_decoder_activation,
            reaction_cluster_decoder_output_size=params.cluster_decoder_projection_head_size,
        )

        # reaction cluster functions
        self.reaction_cluster_fn = {mode: None for mode in ["train", "val", "test"]}
        self.assignments = {mode: None for mode in ["train", "val", "test"]}
        self.centroids = {mode: None for mode in ["train", "val", "test"]}

        # metrics
        self.metrics = self._init_metrics()

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

    def on_train_epoch_start(self):
        if self.reaction_cluster_fn["train"] is None:
            self.reaction_cluster_fn["train"] = self._init_reaction_cluster_fn(
                self.train_dataloader()
            )

        self._compute_reaction_cluster_assignments("train")

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

        self._compute_reaction_cluster_assignments("val")

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

        self._compute_reaction_cluster_assignments("test")

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
        cluster_assignments = self.assignments[mode]
        cluster_centroids = self.centroids[mode]
        loss_reaction_cluster = []
        for a, c in zip(cluster_assignments, cluster_centroids):
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

    def _compute_reaction_cluster_assignments(self, mode):
        """
        cluster the reactions based on reaction features after mapping head
        """
        assi, cent = self.reaction_cluster_fn[mode].get_cluster_assignments()
        self.assignments[mode] = assi
        self.centroids[mode] = cent

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

    def _update_metrics(self, preds, labels, mode):
        """
        update metric states at each step
        """
        mode = "metric_" + mode

        keys = ["bond_hop_dist", "atom_hop_dist", "masked_atom_type"]
        for key in keys:
            for mt in self.metrics[mode][key]:
                metric_obj = self.metrics[mode][key][mt]
                metric_obj(preds[key], labels[key])

    def _compute_metrics(self, mode):
        """
        compute metric and log it at each epoch
        """
        mode = "metric_" + mode

        sum_f1 = 0
        keys = ["bond_hop_dist", "atom_hop_dist", "masked_atom_type"]
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
