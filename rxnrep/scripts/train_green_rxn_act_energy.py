"""
Decoders:
- reaction energy
- activation energy
"""

import argparse
from datetime import datetime

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from rxnrep.model.model import ReactionRepresentation
from rxnrep.scripts import argument
from rxnrep.scripts.load_dataset import load_Green_dataset
from rxnrep.scripts.main import main
from rxnrep.scripts.utils import TimeMeter, get_repo_git_commit


def parse_args():
    parser = argparse.ArgumentParser(description="Reaction Representation")

    # ========== dataset ==========
    prefix = "/Users/mjwen/Documents/Dataset/activation_energy_Green/"

    fname_tr = prefix + "wb97xd3_n200_processed_train.tsv"
    fname_val = fname_tr
    fname_test = fname_tr

    parser.add_argument("--trainset_filename", type=str, default=fname_tr)
    parser.add_argument("--valset_filename", type=str, default=fname_val)
    parser.add_argument("--testset_filename", type=str, default=fname_test)
    parser.add_argument(
        "--dataset_state_dict_filename", type=str, default="dataset_state_dict.yaml"
    )

    # ========== model ==========
    parser = argument.encoder_args(parser)
    parser = argument.reaction_energy_decoder_args(parser)
    parser = argument.activation_energy_decoder_args(parser)

    # ========== training ==========
    parser = argument.training_args(parser)

    # ========== helper ==========
    parser = argument.encoder_helper(parser)
    parser = argument.energy_decoder_helper(parser)

    ####################
    args = parser.parse_args()
    ####################

    # ========== adjuster ==========
    args = argument.encoder_adjuster(args)
    args = argument.reaction_energy_decoder_adjuster(args)
    args = argument.activation_energy_decoder_adjuster(args)

    return args


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
            # compressing
            compressing_layer_sizes=params.compressing_layer_sizes,
            compressing_layer_activation=params.compressing_layer_activation,
            # pooling method
            pooling_method=params.pooling_method,
            pooling_kwargs=params.pooling_kwargs,
            # energy decoder
            reaction_energy_decoder_hidden_layer_sizes=params.reaction_energy_decoder_hidden_layer_sizes,
            reaction_energy_decoder_activation=params.reaction_energy_decoder_activation,
            activation_energy_decoder_hidden_layer_sizes=params.activation_energy_decoder_hidden_layer_sizes,
            activation_energy_decoder_activation=params.activation_energy_decoder_activation,
        )

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
            If returns = `activation_energy` (`reaction_energy`), return the activation
            (reaction) energy predicted by the decoder.
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
        elif returns in [
            "reaction_energy",
            "activation_energy",
        ]:
            feats, reaction_feats = self.model(mol_graphs, rxn_graphs, feats, metadata)
            preds = self.model.decode(feats, reaction_feats, metadata)

            mean = self.hparams.label_mean[returns]
            std = self.hparams.label_std[returns]
            preds = preds[returns] * std + mean

            return preds

        else:
            supported = [
                "reaction_feature",
                "diff_feature_before_rxn_conv",
                "diff_feature_after_rxn_conv",
                "reaction_energy",
                "activation_energy",
            ]
            raise ValueError(
                f"Expect `returns` to be one of {supported}; got `{returns}`."
            )

    def training_step(self, batch, batch_idx):
        loss, preds, labels, indices = self.shared_step(batch, "train")
        self._update_metrics(preds, labels, "train")

        return {"loss": loss}

    def training_epoch_end(self, outputs):
        self._compute_metrics("train")

    def validation_step(self, batch, batch_idx):
        loss, preds, labels, indices = self.shared_step(batch, "val")
        self._update_metrics(preds, labels, "val")

        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        # sum f1 used for early stopping and learning rate scheduler
        sum_f1 = self._compute_metrics("val")

        self.log(f"val/f1", sum_f1, on_step=False, on_epoch=True, prog_bar=True)

        # time it
        delta_t, cumulative_t = self.timer.update()
        self.log("epoch time", delta_t, on_step=False, on_epoch=True, prog_bar=True)
        self.log("cumulative time", cumulative_t, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss, preds, labels, indices = self.shared_step(batch, "test")
        self._update_metrics(preds, labels, "test")

        return {"loss": loss}

    def test_epoch_end(self, outputs):
        self._compute_metrics("test")

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

        #
        # energy loss
        #
        preds["reaction_energy"] = preds["reaction_energy"].flatten()
        loss_reaction_energy = F.mse_loss(
            preds["reaction_energy"], labels["reaction_energy"]
        )

        preds["activation_energy"] = preds["activation_energy"].flatten()
        loss_activation_energy = F.mse_loss(
            preds["activation_energy"], labels["activation_energy"]
        )

        # total loss (maybe assign different weights)
        loss = loss_reaction_energy + loss_activation_energy

        # ========== log the loss ==========
        self.log_dict(
            {
                f"{mode}/loss/reaction_energy": loss_reaction_energy,
                f"{mode}/loss/activation_energy": loss_activation_energy,
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
            optimizer, mode="max", factor=0.4, patience=50, verbose=True
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val/f1"}

    def _init_metrics(self):
        # should be modules so that metric tensors can be placed in the correct device

        metrics = nn.ModuleDict()

        for mode in ["metric_train", "metric_val", "metric_test"]:

            metrics[mode] = nn.ModuleDict()

            for key in [
                "reaction_energy",
                "activation_energy",
            ]:
                metrics[mode][key] = nn.ModuleDict(
                    {"mae": pl.metrics.MeanAbsoluteError(compute_on_step=False)}
                )

        return metrics

    def _update_metrics(
        self,
        preds,
        labels,
        mode,
        keys=(
            "reaction_energy",
            "activation_energy",
        ),
    ):
        """
        update metric states at each step.
        """
        mode = "metric_" + mode

        for key in keys:
            for name in self.metrics[mode][key]:
                metric_obj = self.metrics[mode][key][name]
                metric_obj(preds[key], labels[key])

    def _compute_metrics(
        self,
        mode,
        keys=(
            "reaction_energy",
            "activation_energy",
        ),
        label_scaler={
            "reaction_energy": "reaction_energy",
            "activation_energy": "activation_energy",
        },
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

                # scale mae labels
                if key in label_scaler and name == "mae":
                    value *= self.hparams.label_std[label_scaler[key]].to(self.device)

                self.log(
                    f"{mode}/{name}/{key}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )

                # reset is called automatically somewhere in lightning, here we call it
                # explicitly just in case
                metric_obj.reset()

                if name == "f1":
                    sum_f1 += value
                # NOTE, we abuse the sum_f1 to add the mae of reaction energy
                # prediction as well
                elif name == "mae":
                    sum_f1 -= value

        return sum_f1


if __name__ == "__main__":

    repo_path = "/Users/mjwen/Applications/rxnrep"
    latest_commit = get_repo_git_commit(repo_path)
    print("Git commit:\n", latest_commit)

    print("Start training at:", datetime.now())

    pl.seed_everything(25)

    # args
    args = parse_args()

    # dataset
    train_loader, val_loader, test_loader = load_Green_dataset(args)

    # model
    model = RxnRepLightningModel(args)

    project = "tmp-rxnrep"
    main(args, model, train_loader, val_loader, test_loader, project)

    print("Finish training at:", datetime.now())
