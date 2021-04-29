import argparse
import logging
from datetime import datetime

import pytorch_lightning as pl
import torch.nn.functional as F

from rxnrep.model.model import ReactionRepresentation
from rxnrep.scripts.load_predictive_dataset import load_dataset
from rxnrep.scripts.utils import write_running_metadata
from rxnrep.scripts import argument
from rxnrep.scripts.base_lit_model import BaseLightningModel
from rxnrep.scripts.main import main

logger = logging.getLogger(__name__)


def parse_args(dataset):
    parser = argparse.ArgumentParser()

    # ========== dataset ==========
    parser = argument.dataset_args(parser, dataset)

    # ========== model ==========
    parser = argument.general_args(parser)
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


class LightningModel(BaseLightningModel):
    def init_backbone(self, params):

        model = ReactionRepresentation(
            in_feats=params.feature_size,
            embedding_size=params.embedding_size,
            # encoder
            molecule_conv_layer_sizes=params.molecule_conv_layer_sizes,
            molecule_num_fc_layers=params.molecule_num_fc_layers,
            molecule_batch_norm=params.molecule_batch_norm,
            molecule_activation=params.activation,
            molecule_residual=params.molecule_residual,
            molecule_dropout=params.molecule_dropout,
            reaction_conv_layer_sizes=params.reaction_conv_layer_sizes,
            reaction_num_fc_layers=params.reaction_num_fc_layers,
            reaction_batch_norm=params.reaction_batch_norm,
            reaction_activation=params.activation,
            reaction_residual=params.reaction_residual,
            reaction_dropout=params.reaction_dropout,
            #
            conv=params.conv,
            has_global_feats=params.has_global_feats,
            # mlp diff
            mlp_diff_layer_sizes=params.mlp_diff_layer_sizes,
            mlp_diff_layer_batch_norm=params.mlp_diff_layer_batch_norm,
            mlp_diff_layer_activation=params.activation,
            # pool method
            pool_method=params.pool_method,
            pool_atom_feats=params.pool_atom_feats,
            pool_bond_feats=params.pool_bond_feats,
            pool_global_feats=params.pool_global_feats,
            pool_kwargs=params.pool_kwargs,
            # mlp pool
            mlp_pool_layer_sizes=params.mlp_pool_layer_sizes,
            mlp_pool_layer_batch_norm=params.mlp_pool_layer_batch_norm,
            mlp_pool_layer_activation=params.activation,
            # energy decoder
            reaction_energy_decoder_hidden_layer_sizes=params.reaction_energy_decoder_hidden_layer_sizes,
            reaction_energy_decoder_activation=params.activation,
            activation_energy_decoder_hidden_layer_sizes=params.activation_energy_decoder_hidden_layer_sizes,
            activation_energy_decoder_activation=params.activation,
        )

        return model

    def init_tasks(self):
        self.regression_tasks = {
            "reaction_energy": {
                "label_scaler": "reaction_energy",
                "to_score": [],
            },
            "activation_energy": {
                "label_scaler": "activation_energy",
                "to_score": {"mae": -1},
            },
        }

    def decode(self, feats, reaction_feats, metadata):
        return self.backbone.decode(feats, reaction_feats, metadata)

    def compute_loss(self, preds, labels):

        all_loss = {}

        # reaction energy loss
        task = "reaction_energy"
        preds[task] = preds[task].flatten()
        loss = F.mse_loss(preds[task], labels[task])
        all_loss[task] = loss

        # activation energy loss
        task = "activation_energy"
        preds[task] = preds[task].flatten()
        loss = F.mse_loss(preds[task], labels[task])
        all_loss[task] = loss

        return all_loss


if __name__ == "__main__":

    logger.info(f"Start training at: {datetime.now()}")
    pl.seed_everything(25)

    filename = "running_metadata.yaml"
    repo_path = "/Users/mjwen/Applications/rxnrep"
    write_running_metadata(filename, repo_path)

    # args
    dataset = "green"
    args = parse_args(dataset)
    logger.info(args)

    # dataset
    train_loader, val_loader, test_loader = load_dataset(args)

    # model
    model = LightningModel(args)

    project = "tmp-rxnrep"
    main(args, model, train_loader, val_loader, test_loader, __file__, project=project)

    logger.info(f"Finish training at: {datetime.now()}")
