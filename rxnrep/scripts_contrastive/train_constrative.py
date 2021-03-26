import argparse
import logging
from datetime import datetime

import pytorch_lightning as pl
from torch.nn import functional as F

from rxnrep.model.encoder import ReactionEncoder
from rxnrep.model.utils import MLP
from rxnrep.scripts.utils import write_running_metadata
from rxnrep.scripts_contrastive import argument
from rxnrep.scripts_contrastive.base_contrastive_lit_model import BaseLightningModel
from rxnrep.scripts_contrastive.load_dataset import load_dataset
from rxnrep.scripts_contrastive.losses import nt_xent_loss
from rxnrep.scripts_contrastive.main import main

logger = logging.getLogger(__name__)


def parse_args(dataset):
    parser = argparse.ArgumentParser()

    # ========== dataset ==========
    parser = argument.dataset_args(parser, dataset)
    parser = argument.data_augmentation_args(parser)

    # ========== model ==========
    parser = argument.general_args(parser)
    parser = argument.encoder_args(parser)
    parser = argument.simclr_decoder_args(parser)

    # ========== training ==========
    parser = argument.training_args(parser)

    # ========== helper ==========
    parser = argument.encoder_helper(parser)
    parser = argument.simclr_decoder_helper(parser)

    ####################
    args = parser.parse_args()
    ####################

    # ========== adjuster ==========
    args = argument.encoder_adjuster(args)
    args = argument.simclr_decoder_adjuster(args)

    return args


class LightningModel(BaseLightningModel):
    def init_model(self, params):

        model = ReactionEncoder(
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
            pool_kwargs=params.pool_kwargs,
            # mlp pool
            mlp_pool_layer_sizes=params.mlp_pool_layer_sizes,
            mlp_pool_layer_batch_norm=params.mlp_pool_layer_batch_norm,
            mlp_pool_layer_activation=params.activation,
        )

        #
        # decoder
        #
        # name this `.._decoder` so that we can easily freeze it when finetune
        self.projection_decoder = MLP(
            in_size=model.reaction_feats_size,
            hidden_sizes=params.simclr_hidden_layer_sizes,
            activation=params.activation,
        )

        return model

    def init_tasks(self):
        pass

    def decode(self, feats, reaction_feats, metadata):
        z = self.projection_decoder(reaction_feats)

        return z

    def compute_loss(self, preds, labels):
        z1 = preds["z1"]
        z2 = preds["z2"]

        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        loss = nt_xent_loss(z1, z2, self.hparams.simclr_temperature)

        return {"contrastive": loss}


if __name__ == "__main__":

    logger.info(f"Start training at: {datetime.now()}")
    pl.seed_everything(25)

    filename = "running_metadata.yaml"
    repo_path = "/Users/mjwen/Applications/rxnrep"
    write_running_metadata(filename, repo_path)

    # args
    dataset = "schneider"
    args = parse_args(dataset)
    logger.info(args)

    # dataset
    train_loader, val_loader, test_loader = load_dataset(args)

    # model
    model = LightningModel(args)

    project = "tmp-rxnrep"
    main(
        args,
        model,
        train_loader,
        val_loader,
        test_loader,
        __file__,
        monitor="val/loss",
        monitor_mode="min",
        project=project,
        run_test=False,
    )

    logger.info(f"Finish training at: {datetime.now()}")
