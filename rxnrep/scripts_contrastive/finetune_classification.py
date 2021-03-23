import argparse
import logging
from datetime import datetime

import pytorch_lightning as pl
import torch.nn.functional as F

from rxnrep.model.utils import MLP
from rxnrep.scripts.load_dataset import load_dataset
from rxnrep.scripts.utils import (
    copy_trained_model,
    load_lightning_pretrained_model,
    write_running_metadata,
)
from rxnrep.scripts_contrastive import argument
from rxnrep.scripts_contrastive.base_lit_model import BaseLightningModel
from rxnrep.scripts_contrastive.main import main
from rxnrep.scripts_contrastive.train_constrative import (
    LightningModel as PretrainedModel,
)

logger = logging.getLogger(__name__)


def parse_args(dataset):
    parser = argparse.ArgumentParser()

    # ========== dataset ==========
    parser = argument.dataset_args(parser, dataset)

    # ========== model ==========
    parser = argument.encoder_args(parser)
    parser = argument.reaction_type_decoder_args(parser)
    parser = argument.finetune_args(parser)

    # ========== training ==========
    parser = argument.training_args(parser)

    # ========== helper ==========
    parser = argument.encoder_helper(parser)
    parser = argument.reaction_type_decoder_helper(parser)

    ####################
    args = parser.parse_args()
    ####################

    # ========== adjuster ==========
    args = argument.encoder_adjuster(args)
    args = argument.reaction_type_decoder_adjuster(args)

    return args


class LightningModel(BaseLightningModel):
    def init_model(self, params):
        #
        # backbone model
        #
        model = load_lightning_pretrained_model(
            PretrainedModel, params.pretrained_ckpt_path
        )

        # select parameters to freeze
        if params.pretrained_tune_encoder:
            # only fix parameters in the decoder
            for name, p in model.named_parameters():
                if "decoder" in name:
                    p.requires_grad = False
        else:
            # fix all backbone parameters
            model.freeze()

        #
        # decoder to predict classes
        #
        # model.model is the model created in init_model() of the pretrained model
        self.mlp = MLP(
            in_size=model.model.reaction_feats_size,
            hidden_sizes=params.reaction_type_decoder_hidden_layer_sizes,
            activation=params.activation,
            out_size=params.num_reaction_classes,
        )

        return model

    def init_tasks(self):
        self.classification_tasks = {
            "reaction_type": {
                "num_classes": self.hparams.num_reaction_classes,
                "to_score": {"f1": 1},
            }
        }

    def decode(self, feats, reaction_feats, metadata):
        preds = {"reaction_type": self.mlp(reaction_feats)}
        return preds

    def compute_loss(self, preds, labels):

        loss = F.cross_entropy(
            preds["reaction_type"],
            labels["reaction_type"],
            reduction="mean",
            weight=self.hparams.reaction_class_weight.to(self.device),
        )

        return {"reaction_type": loss}

    def on_train_epoch_start(self):
        # Although model.eval() is called in mode.freeze() when calling init_model(),
        # we call it explicitly at each train epoch in case lightning calls
        # self.train() internally to change the states of dropout and batch norm
        if not self.hparams.pretrained_tune_encoder:
            self.model.eval()


if __name__ == "__main__":

    logger.info(f"Start training at: {datetime.now()}")
    pl.seed_everything(25)

    filename = "running_metadata.yaml"
    repo_path = "/Users/mjwen/Applications/rxnrep"
    write_running_metadata(filename, repo_path)

    # args
    dataset = "schneider_classification"
    args = parse_args(dataset)
    logger.info(args)

    #
    # pretrained model info
    #
    pretrained_model_identifier = "3fxgra68"
    target_dir = "pretrained_model"
    copy_trained_model(pretrained_model_identifier, target_dir=target_dir)

    # dataset
    train_loader, val_loader, test_loader = load_dataset(args)

    # model
    model = LightningModel(args)

    project = "tmp-rxnrep"
    main(args, model, train_loader, val_loader, test_loader, __file__, project=project)

    logger.info(f"Finish training at: {datetime.now()}")
