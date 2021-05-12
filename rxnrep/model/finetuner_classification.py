"""
Finetune simclr pretrained model.
"""

import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from rxnrep.layer.utils import MLP
from rxnrep.model.model_finetune import BaseFinetuneModel
from rxnrep.model.simclr import LightningModel as PretrainedModel
from rxnrep.utils.config import (
    determine_layer_size_by_pool_method,
    get_datamodule_config,
    merge_configs,
)


def get_pretrained_model_config(config: DictConfig) -> DictConfig:
    """
    Get pretrained model info that will be used by decoder adjuster.

    We can simply return the whole pretrain model config, but this is prone to error.
    Instead, we return a new config only including the info we want.
    """
    pretrained_cfg = OmegaConf.load(
        config.model.finetuner.model_class.pretrained_config_filename
    )

    keys = [
        "conv_layer_size",
        "pool_method",
        "pool_atom_feats",
        "pool_bond_feats",
        "pool_global_feats",
    ]
    d = {k: pretrained_cfg.model.encoder[k] for k in keys}

    cfg = OmegaConf.create({"model": {"encoder": d}})

    return cfg


def adjust_datamodule_config(config: DictConfig) -> DictConfig:
    """
    Set restore_state_dict_filename to that of the pretrained model.
    """
    dm_config, name = get_datamodule_config(config)

    filename = config.model.finetuner.model_class.pretrained_dataset_state_dict_filename
    cfg = OmegaConf.create(
        {"datamodule": {name: {"restore_state_dict_filename": filename}}}
    )

    return cfg


def adjust_decoder_config(config: DictConfig):
    size = determine_layer_size_by_pool_method(config.model.encoder)

    num_layers = config.model.finetuner.model_class.reaction_type_decoder_num_layers
    hidden_layer_sizes = [max(size // 2 ** i, 50) for i in range(num_layers)]

    new_config = OmegaConf.create(
        {
            "model": {
                "finetuner": {
                    "model_class": {
                        "reaction_type_decoder_hidden_layer_sizes": hidden_layer_sizes
                    }
                }
            }
        }
    )

    return new_config


def adjust_config(config: DictConfig) -> DictConfig:
    """
    Adjust model config, only need to adjust decoder.
    """
    dm_cfg = adjust_datamodule_config(config)

    info_cfg = get_pretrained_model_config(config)
    merged = merge_configs(config, info_cfg)
    decoder_cfg = adjust_decoder_config(merged)

    return merge_configs(dm_cfg, decoder_cfg)


class LightningModel(BaseFinetuneModel):
    def init_backbone(self, params):
        model = PretrainedModel.load_from_checkpoint(
            params.pretrained_checkpoint_filename
        )

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
            hidden_sizes=params.reaction_type_decoder_hidden_layer_sizes,
            activation=params.activation,
            out_size=params.dataset_info["num_reaction_classes"],
        )

        return {"reaction_type_decoder": decoder}

    def decode(self, feats, reaction_feats, metadata):
        decoder = self.decoder["reaction_type_decoder"]
        reaction_type = decoder(reaction_feats)

        return {"reaction_type": reaction_type}

    def compute_loss(self, preds, labels):

        all_loss = {}

        task = "reaction_type"
        loss = F.cross_entropy(
            preds[task],
            labels[task],
            reduction="mean",
            weight=self.hparams.dataset_info["reaction_class_weight"].to(self.device),
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

    def on_train_epoch_start(self):
        # Although model.eval() is called in mode.freeze() when calling init_backbone(),
        # we call it explicitly at each train epoch in case lightning calls
        # self.train() internally to change the states of dropout and batch norm
        if not self.hparams.finetune_tune_encoder:
            self.backbone.eval()
