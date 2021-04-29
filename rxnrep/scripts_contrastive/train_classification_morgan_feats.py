import argparse
import logging
from datetime import datetime

import pytorch_lightning as pl
import torch.nn.functional as F

from rxnrep.model.utils import MLP
from rxnrep.scripts.load_dataset import load_morgan_feature_dataset
from rxnrep.scripts.utils import write_running_metadata
from rxnrep.scripts_contrastive import argument
from rxnrep.scripts_contrastive.base_lit_model import BaseLightningModel
from rxnrep.scripts_contrastive.cross_validate import cross_validate
from rxnrep.scripts_contrastive.main import main

logger = logging.getLogger(__name__)


def parse_args(dataset):
    parser = argparse.ArgumentParser()

    # ========== dataset ==========
    parser = argument.dataset_args(parser, dataset)

    # encoder
    parser = argument.general_args(parser)
    parser.add_argument("--morgan_radius", type=int, default=2)
    parser.add_argument("--morgan_size", type=int, default=2048)
    parser.add_argument("--feature_pool_type", type=str, default="difference")

    # decoder
    parser = argument.reaction_type_decoder_args(parser)
    parser = argument.reaction_type_decoder_helper(parser)

    # training
    parser = argument.training_args(parser)

    ###############
    args = parser.parse_args()
    ###############

    # adjuster
    if args.feature_pool_type == "difference":
        val = args.morgan_size
    elif args.feature_pool_type == "concatenate":
        val = 2 * args.morgan_size
    else:
        raise ValueError
    args.reaction_type_decoder_hidden_layer_sizes = [
        max(val // 2 ** i, 50) for i in range(args.reaction_type_decoder_num_layers)
    ]

    return args


class LightningModel(BaseLightningModel):
    def init_backbone(self, params):

        self.reaction_type_decoder = MLP(
            in_size=params.reaction_type_decoder_hidden_layer_sizes[0],
            hidden_sizes=params.reaction_type_decoder_hidden_layer_sizes,
            activation=params.activation,
            out_size=params.num_reaction_classes,
        )

    def init_tasks(self):
        self.classification_tasks = {
            "reaction_type": {
                "num_classes": self.hparams.num_reaction_classes,
                "to_score": {"f1": 1},
            }
        }

    def shared_step(self, batch, mode):
        # ========== compute predictions ==========
        feats, labels = batch
        preds = {"reaction_type": self.reaction_type_decoder(feats)}

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

        return total_loss, preds, labels, None

    def compute_loss(self, preds, labels):

        loss = F.cross_entropy(
            preds["reaction_type"],
            labels["reaction_type"],
            reduction="mean",
            # weight=self.hparams.reaction_class_weight.to(self.device),
        )

        return {"reaction_type": loss}


if __name__ == "__main__":

    logger.info(f"Start training at: {datetime.now()}")
    pl.seed_everything(25)

    filename = "running_metadata.yaml"
    repo_path = "/Users/mjwen/Applications/rxnrep"
    write_running_metadata(filename, repo_path)

    # args
    # dataset = "schneider_classification"
    dataset = "green_classification"
    args = parse_args(dataset)
    logger.info(args)
    # args.num_reaction_classes = 1000

    project = "tmp-rxnrep"

    if args.kfold:
        cross_validate(
            args,
            LightningModel,
            load_morgan_feature_dataset,
            main,
            stratify_column="reaction_type",
            fold=args.kfold,
            project=project,
        )

    else:
        # dataset
        train_loader, val_loader, test_loader = load_morgan_feature_dataset(args)

        # model
        model = LightningModel(args)

        main(
            args,
            model,
            train_loader,
            val_loader,
            test_loader,
            __file__,
            project=project,
        )

    logger.info(f"Finish training at: {datetime.now()}")
