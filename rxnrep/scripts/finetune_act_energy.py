import argparse
import logging
from datetime import datetime

import pytorch_lightning as pl
import torch.nn.functional as F

from rxnrep.model.model_finetune import BaseFinetuneModel
from rxnrep.model.utils import MLP
from rxnrep.scripts import argument
from rxnrep.scripts.load_predictive_dataset import load_dataset
from rxnrep.scripts.main import main
from rxnrep.scripts.train_simclr import LightningModel as PretrainedModel
from rxnrep.scripts.utils import copy_trained_model, write_running_metadata
from rxnrep.utils import yaml_load

logger = logging.getLogger(__name__)


def parse_args(dataset):
    parser = argparse.ArgumentParser()

    # ========== dataset ==========
    parser = argument.dataset_args(parser, dataset)

    # ========== model ==========
    parser = argument.general_args(parser)
    parser = argument.activation_energy_decoder_args(parser)
    parser = argument.finetune_args(parser)

    # ========== training ==========
    parser = argument.training_args(parser)

    # ========== helper ==========
    parser = argument.energy_decoder_helper(parser)

    ####################
    args = parser.parse_args()
    ####################

    # TODO replace all these by backbone.reaction_feats_size
    #
    # set `--conv_layer_size` to be the value used in pretrained model, which will be
    # used in many adjusters. Essentially, we can extract this info from the pretrained
    # model `model.reaction_feats_size`, but here we just extract it from the running
    # info of the pretrained model.
    #
    # conv_layer_size: determine prediction head size
    # pool_method, pool_atom_feats, pool_bond_feats, pool_global_feats:
    # determine prediction head size
    # reaction_conv_layer_sizes: determine whether to build reaction graphs (used in
    # load_dataset)
    d = yaml_load(args.pretrained_config_filename)
    args.conv_layer_size = d["conv_layer_size"]["value"]
    args.pool_method = d["pool_method"]["value"]
    args.pool_atom_feats = d["pool_atom_feats"]["value"]
    args.pool_bond_feats = d["pool_bond_feats"]["value"]
    args.pool_global_feats = d["pool_global_feats"]["value"]
    args.reaction_conv_layer_sizes = d["reaction_conv_layer_sizes"]["value"]

    # ========== adjuster ==========
    args = argument.activation_energy_decoder_adjuster(args)

    # do not require pretrained model to have mean and std for labels
    args.allow_label_scaler_none = 1

    return args


class LightningModel(BaseFinetuneModel):
    def init_backbone(self, params):
        model = PretrainedModel.load_from_checkpoint(params.pretrained_ckpt_path)

        # select parameters to freeze
        if params.finetune_tune_encoder:
            # only fix parameters in the decoder
            for name, p in model.named_parameters():
                if "decoder" in name:
                    p.requires_grad = False
        else:
            # fix all backbone parameters
            model.freeze()

        return model

    def init_decoder(self, params):
        decoder = MLP(
            in_size=self.backbone.backbone.reaction_feats_size,
            hidden_sizes=params.activation_energy_decoder_hidden_layer_sizes,
            activation=params.activation,
            out_size=1,
        )

        return {"activation_energy_decoder": decoder}

    def decode(self, feats, reaction_feats, metadata):
        decoder = self.decoder["activation_energy_decoder"]
        activation_energy = decoder(reaction_feats)

        return {"activation_energy": activation_energy}

    def compute_loss(self, preds, labels):

        all_loss = {}

        task = "activation_energy"
        preds[task] = preds[task].flatten()
        loss = F.mse_loss(preds[task], labels[task])
        all_loss[task] = loss

        return all_loss

    def init_regression_tasks(self, params):
        tasks = {
            "activation_energy": {
                "label_scaler": "activation_energy",
                "to_score": {"mae": -1},
            }
        }

        return tasks

    def on_train_epoch_start(self):
        # Although model.eval() is called in mode.freeze() when calling init_backbone(),
        # we call it explicitly at each train epoch in case lightning calls
        # self.train() internally to change the states of dropout and batch norm
        if not self.hparams.finetune_tune_encoder:
            self.backbone.eval()


if __name__ == "__main__":

    logger.info(f"Start training at: {datetime.now()}")
    pl.seed_everything(25)

    filename = "running_metadata.yaml"
    repo_path = "/Users/mjwen/Applications/rxnrep"
    write_running_metadata(filename, repo_path)

    #
    # pretrained model info
    #
    pretrained_model_identifier = "1ya7egg6"
    target_dir = "pretrained_model"
    copy_trained_model(pretrained_model_identifier, target_dir=target_dir)

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
