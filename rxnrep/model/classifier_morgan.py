"""
Classifier using morgan feats.
"""
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from rxnrep.layer.utils import MLP
from rxnrep.model.model import BaseModel
from rxnrep.utils.config import get_datamodule_config


def adjust_config(config: DictConfig):
    dm_config, _ = get_datamodule_config(config)
    size = dm_config["morgan_size"]

    num_layers = config.model.decoder.model_class.reaction_type_decoder_num_layers
    hidden_layer_sizes = [max(size // 2 ** i, 50) for i in range(num_layers)]

    new_config = OmegaConf.create(
        {
            "model": {
                "decoder": {
                    "model_class": {
                        "reaction_type_decoder_hidden_layer_sizes": hidden_layer_sizes
                    }
                }
            }
        }
    )

    return new_config


class LightningModel(BaseModel):
    def init_backbone(self, params):
        return None

    def init_decoder(self, params):
        num_reaction_classes = params.dataset_info["num_reaction_classes"]

        if num_reaction_classes == 2:
            self.is_binary = True
            out_size = 1
        else:
            self.is_binary = False
            out_size = num_reaction_classes

        decoder = MLP(
            in_size=params.reaction_type_decoder_hidden_layer_sizes[0],
            hidden_sizes=params.reaction_type_decoder_hidden_layer_sizes,
            activation=params.activation,
            out_size=out_size,
        )

        return {"reaction_type_decoder": decoder}

    def shared_step(self, batch, mode):

        decoder = self.decoder["reaction_type_decoder"]
        # ========== compute predictions ==========
        feats, labels = batch
        preds = {"reaction_type": decoder(feats)}

        # ========== compute losses ==========
        all_loss = self.compute_loss(preds, labels)

        # ========== logger the loss ==========
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

        all_loss = {}

        task = "reaction_type"

        if self.is_binary:
            loss = F.binary_cross_entropy_with_logits(
                preds[task].reshape(-1),
                labels[task].to(torch.float),  # input label is int for metric purpose
                reduction="mean",
            )
        else:
            loss = F.cross_entropy(
                preds[task],
                labels[task],
                reduction="mean",
                weight=self.hparams.dataset_info["reaction_class_weight"].to(
                    self.device
                ),
            )

        all_loss[task] = loss

        return all_loss

    def init_classification_tasks(self, params):
        tasks = {
            "reaction_type": {
                "num_classes": params.dataset_info["num_reaction_classes"],
                "to_score": {"f1": 1},
                "average": "micro",
            }
        }

        return tasks
