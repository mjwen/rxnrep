"""
Base Lightning model for regression and classification.
"""

from typing import Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim import lr_scheduler

from rxnrep.scripts.utils import TimeMeter


class BaseLightningModel(pl.LightningModule):
    def __init__(self, params):
        super().__init__()

        self.save_hyperparameters(params)

        self.backbone = self.init_backbone(self.hparams)

        self.classification_tasks = {}
        self.regression_tasks = {}
        self.init_tasks()

        self.metrics = self.init_metrics()

        self.timer = TimeMeter()

    def forward(
        self, mol_graphs, rxn_graphs, feats, metadata, return_mode: Optional[str] = None
    ):
        """
        Args:
            return_mode: select what to return. See below.

        Returns:
            If `None`, directly return the value returned by the self.model forward
            method. This is typically the features and reaction features returned
            by the decoder, e.g. (feats, reaction_feats). feats is a dictionary of
            atom, bond, and global features and reaction_feats is the reaction
            feature tensor.
            Values from the decoder can also be returned; currently supported ones are
            `reaction_energy`, `activation_energy`, and `reaction_type`.
        """

        if return_mode is None:
            return self.backbone(mol_graphs, rxn_graphs, feats, metadata)

        elif return_mode == "difference_feature":
            return self.get_difference_feature(mol_graphs, rxn_graphs, feats, metadata)

        elif return_mode == "pool_attention_score":
            return self.get_pool_attention_score(
                mol_graphs, rxn_graphs, feats, metadata
            )

        elif return_mode in ["reaction_energy", "activation_energy"]:  # regression
            feats, reaction_feats = self.backbone(
                mol_graphs, rxn_graphs, feats, metadata
            )
            preds = self.decode(feats, reaction_feats, metadata)

            state_dict = self.hparams.label_scaler[return_mode].state_dict()
            mean = state_dict["mean"]
            std = state_dict["std"]
            preds = preds[return_mode] * std + mean

            return preds

        elif return_mode == "reaction_type":  # classification
            feats, reaction_feats = self.backbone(
                mol_graphs, rxn_graphs, feats, metadata
            )
            preds = self.decode(feats, reaction_feats, metadata)
            return preds[return_mode]

        else:
            supported = [
                None,
                "difference_feature",
                "poll_attention_score",
                "reaction_energy",
                "activation_energy",
                "reaction_type",
            ]
            raise ValueError(
                f"Expect `return_mode` to be one of {supported}; got `{return_mode}`."
            )

    def training_step(self, batch, batch_idx):
        loss, preds, labels, indices = self.shared_step(batch, "train")
        self.update_metrics(preds, labels, "train")

        return {"loss": loss}

    def training_epoch_end(self, outputs):
        self.compute_metrics("train")

    def validation_step(self, batch, batch_idx):
        loss, preds, labels, indices = self.shared_step(batch, "val")
        self.update_metrics(preds, labels, "val")

        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        score = self.compute_metrics("val")

        # val/score used for early stopping and learning rate scheduler
        if score is not None:
            self.log(f"val/score", score, on_step=False, on_epoch=True, prog_bar=True)

        # time it
        delta_t, cumulative_t = self.timer.update()
        self.log("epoch time", delta_t, on_step=False, on_epoch=True, prog_bar=True)
        self.log("cumulative time", cumulative_t, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss, preds, labels, indices = self.shared_step(batch, "test")
        self.update_metrics(preds, labels, "test")

        return {"loss": loss}

    def test_epoch_end(self, outputs):
        self.compute_metrics("test")

    def shared_step(self, batch, mode):

        # ========== compute predictions ==========
        indices, mol_graphs, rxn_graphs, labels, metadata = batch

        # lightning cannot move dgl graphs to gpu, so do it manually
        mol_graphs = mol_graphs.to(self.device)
        if rxn_graphs is not None:
            rxn_graphs = rxn_graphs.to(self.device)

        nodes = ["atom", "global"]
        feats = {nt: mol_graphs.nodes[nt].data.pop("feat") for nt in nodes}
        feats["bond"] = mol_graphs.edges["bond"].data.pop("feat")

        feats, reaction_feats = self(mol_graphs, rxn_graphs, feats, metadata)

        preds = self.decode(feats, reaction_feats, metadata)

        # ========== compute losses ==========
        all_loss = self.compute_loss(preds, labels)

        # ========== log the loss ==========
        total_loss = sum(all_loss.values())

        self.log_dict(
            {f"{mode}/loss/{task}": loss for task, loss in all_loss.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        self.log(
            f"{mode}/loss", total_loss, on_step=False, on_epoch=True, prog_bar=True
        )

        return total_loss, preds, labels, indices

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        if self.hparams.lr_scheduler == "reduce_on_plateau":
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.4, patience=50, verbose=True
            )
        elif self.hparams.lr_scheduler == "cosine":
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=self.hparams.lr_warmup_step,
                max_epochs=self.hparams.epochs,
                eta_min=self.hparams.lr_min,
            )
        else:
            raise ValueError(f"Not supported lr scheduler: {self.hparams.lr_scheduler}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/score",
        }

    def init_metrics(self):
        metrics = nn.ModuleDict()

        for mode in ["metric_train", "metric_val", "metric_test"]:

            metrics[mode] = nn.ModuleDict()

            for task_name, task_setting in self.classification_tasks.items():
                n = task_setting["num_classes"]
                metrics[mode][task_name] = nn.ModuleDict(
                    {
                        "accuracy": pl.metrics.Accuracy(compute_on_step=False),
                        "precision": pl.metrics.Precision(
                            num_classes=n,
                            average="micro",
                            compute_on_step=False,
                        ),
                        "recall": pl.metrics.Recall(
                            num_classes=n,
                            average="micro",
                            compute_on_step=False,
                        ),
                        "f1": pl.metrics.F1(
                            num_classes=n,
                            average="micro",
                            compute_on_step=False,
                        ),
                    }
                )

            for task_name in self.regression_tasks:
                metrics[mode][task_name] = nn.ModuleDict(
                    {"mae": pl.metrics.MeanAbsoluteError(compute_on_step=False)}
                )

        return metrics

    def update_metrics(self, preds, labels, mode):
        """
        Update metric values at each step.
        """
        mode = "metric_" + mode

        # regression metrics
        for task_name in self.regression_tasks:
            for metric in self.metrics[mode][task_name]:
                metric_obj = self.metrics[mode][task_name][metric]
                metric_obj(preds[task_name], labels[task_name])

        # classification metrics
        for task_name in self.classification_tasks:
            for metric in self.metrics[mode][task_name]:
                metric_obj = self.metrics[mode][task_name][metric]
                p = torch.softmax(preds[task_name], dim=1)
                metric_obj(p, labels[task_name])

    def compute_metrics(self, mode):
        """
        Compute metric and log it at each epoch.
        """
        mode = "metric_" + mode

        # Do not set to 0 there in order to return None if there is no metric
        # contributes to score
        score = None

        for task_name, task_setting in self.classification_tasks.items():
            for metric in self.metrics[mode][task_name]:
                metric_obj = self.metrics[mode][task_name][metric]
                out = metric_obj.compute()

                self.log(
                    f"{mode}/{metric}/{task_name}",
                    out,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )

                metric_obj.reset()

                if metric in task_setting["to_score"]:
                    score = 0 if score is None else score
                    sign = task_setting["to_score"][metric]
                    score += out * sign

        for task_name, task_setting in self.regression_tasks.items():
            for metric in self.metrics[mode][task_name]:
                metric_obj = self.metrics[mode][task_name][metric]
                out = metric_obj.compute()

                # scale labels
                label_scaler = task_setting["label_scaler"]
                state_dict = self.hparams.label_scaler[label_scaler].state_dict()
                out *= state_dict["std"].to(self.device)

                self.log(
                    f"{mode}/{metric}/{task_name}",
                    out,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )

                metric_obj.reset()

                if metric in task_setting["to_score"]:
                    score = 0 if score is None else score
                    sign = task_setting["to_score"][metric]
                    score += out * sign

        return score

    def init_backbone(self, params):
        """
        Create backbone model.

        Args:
            params:

        Return:
            The model.

        Example:

        model = ReactionRepresentation(
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
            # mlp_diff
            mlp_diff_layer_sizes=params.mlp_diff_layer_sizes,
            mlp_diff_layer_activation=params.mlp_diff_layer_activation,
            # pool method
            pool_method=params.pool_method,
            pool_kwargs=params.pool_kwargs,
        )

        return model
        """
        raise NotImplementedError

    def init_tasks(self):
        """
        Define the tasks (decoders) and the associated metrics.

        Example:

        self.classification_tasks = {
            # "bond_hop_dist": {
            #     "num_classes": self.hparams.bond_hop_dist_num_classes,
            #     "to_score": {"f1": 1},
            # }
        }

        self.regression_tasks = {
            # "reaction_energy": {
            #     "label_scaler": "reaction_energy",
            #     "to_score": {"mae": -1},
            # }
        }
        """

        raise NotImplementedError

    def decode(
        self,
        feats: Dict[str, torch.Tensor],
        reaction_feats: torch.Tensor,
        metadata: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Decode the molecule and reaction features to properties.

        Args:
            feats: atom and bond (maybe global) features of molecules
            reaction_feats: reaction features
            metadata:

        Returns:
            predictions: {decoder_name: value} predictions of the decoders.
        """

        raise NotImplementedError

    def compute_loss(
        self, preds: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the loss for each task.

        Args:
            preds: {task_name, prediction} prediction for each task
            labels: {task_name, label} labels for each task

        Example:

        all_loss = {}

        # bond hop distance loss
        task = "bond_hop_dist"
        if task in self.classification_tasks:
            loss = F.cross_entropy(
                preds[task],
                labels[task],
                reduction="mean",
                weight=self.hparams.bond_hop_dist_class_weight.to(self.device),
            )
            all_loss[task] = loss

        Returns:
            {task_name, loss}
        """
        raise NotImplementedError

    def get_pool_attention_score(self, mol_graphs, rxn_graphs, feats, metadata):
        """
        Returns:
            Dict of attention score

        """
        attn_score = self.backbone.get_pool_attention_score(
            mol_graphs, rxn_graphs, feats, metadata
        )
        return attn_score

    def get_difference_feature(self, mol_graphs, rxn_graphs, feats, metadata):
        """
        Returns:
            2D tensor of shape (B, D), where B is batch size and D is feature dim.
            Each row for a reaction.
        """
        diff_feat = self.backbone.get_difference_feature(
            mol_graphs, rxn_graphs, feats, metadata
        )
        return diff_feat
