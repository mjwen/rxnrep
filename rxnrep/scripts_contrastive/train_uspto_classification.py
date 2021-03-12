import argparse
import logging
from datetime import datetime

import pytorch_lightning as pl
import torch.nn.functional as F

from rxnrep.model.model import ReactionRepresentation
from rxnrep.scripts.load_dataset import load_uspto_dataset
from rxnrep.scripts.utils import write_running_metadata
from rxnrep.scripts_contrastive import argument
from rxnrep.scripts_contrastive.base_model import BaseLightningModel
from rxnrep.scripts_contrastive.main import main

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    # ========== dataset ==========
    parser.add_argument("--has_class_label", type=int, default=1)

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
    parser = argument.encoder_args(parser)
    parser = argument.reaction_type_decoder_args(parser)

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
    def create_model(self, params):

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
            # compressing
            compressing_layer_sizes=params.compressing_layer_sizes,
            compressing_layer_activation=params.compressing_layer_activation,
            # pooling method
            pooling_method=params.pooling_method,
            pooling_kwargs=params.pooling_kwargs,
            # reaction type decoder
            reaction_type_decoder_hidden_layer_sizes=params.reaction_type_decoder_hidden_layer_sizes,
            reaction_type_decoder_activation=params.reaction_type_decoder_activation,
            reaction_type_decoder_num_classes=params.num_reaction_classes,
        )

        return model

    def init_tasks(self):
        self.classification_tasks = {
            "reaction_type": {
                "num_classes": self.hparams.num_reaction_classes,
                "to_score": {"f1": -1},
            }
        }

    def compute_loss(self, preds, labels):

        loss = F.cross_entropy(
            preds["reaction_type"],
            labels["reaction_type"],
            reduction="mean",
            weight=self.hparams.reaction_class_weight.to(self.device),
        )

        return {"reaction_type": loss}


if __name__ == "__main__":

    logger.info(f"Start training at: {datetime.now()}")

    filename = "running_metadata.yaml"
    repo_path = "/Users/mjwen/Applications/rxnrep"
    write_running_metadata(filename, repo_path)

    pl.seed_everything(25)

    # args
    args = parse_args()
    logger.info(args)

    # dataset
    train_loader, val_loader, test_loader = load_uspto_dataset(args)

    # model
    model = LightningModel(args)

    project = "tmp-rxnrep"
    main(args, model, train_loader, val_loader, test_loader, project)

    logger.info(f"Finish training at: {datetime.now()}")
