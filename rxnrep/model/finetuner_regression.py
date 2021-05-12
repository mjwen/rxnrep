"""
Finetune pretrained model.

The pretrained model can be either a simclr trained or regressed.

Set `PretrainedModel` in the import to determine it.
"""
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from rxnrep.layer.utils import MLP
from rxnrep.model.finetuner_classification import (
    adjust_datamodule_config,
    get_pretrained_model_config,
)
from rxnrep.model.model_finetune import BaseFinetuneModel

# from rxnrep.scripts.train_simclr import LightningModel as PretrainedModel
from rxnrep.model.regressor import LightningModel as PretrainedModel
from rxnrep.utils.config import determine_layer_size_by_pool_method, merge_configs


def adjust_decoder_config(config: DictConfig):
    size = determine_layer_size_by_pool_method(config.model.encoder)

    num_layers = config.model.finetuner.model_class.regression_decoder_num_layers

    hidden_layer_sizes = []
    for n in num_layers:
        hidden_layer_sizes.append([max(size // 2 ** i, 50) for i in range(n)])

    new_config = OmegaConf.create(
        {
            "model": {
                "finetuner": {
                    "model_class": {
                        "regression_decoder_hidden_layer_sizes": hidden_layer_sizes
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
        decoder = {}

        for name, size in zip(
            params.property_name, params.regression_decoder_hidden_layer_sizes
        ):
            name = name + "_decoder"  # e.g. reaction_energy_decoder

            decoder[name] = MLP(
                in_size=self.backbone.backbone.reaction_feats_size,
                hidden_sizes=size,
                activation=params.activation,
                out_size=1,
            )

        return decoder

    def decode(self, feats, reaction_feats, metadata):
        preds = {}

        for name, dec in self.decoder.items():
            name = name.rstrip("_decoder")  # e.g. reaction_energy
            preds[name] = dec(reaction_feats)

        return preds

    def compute_loss(self, preds, labels):

        all_loss = {}

        for task, p in preds.items():
            p = p.flatten()
            t = labels[task]
            loss = F.mse_loss(p, t)
            all_loss[task] = loss

            # keep flattened value for metric use
            preds[task] = p

        return all_loss

    def init_regression_tasks(self, params):
        tasks = {}

        for name in params.property_name:
            tasks[name] = {
                "label_scaler": name,
                "to_score": {"mae": -1},
            }

        return tasks

    def on_train_epoch_start(self):
        # Although model.eval() is called in mode.freeze() when calling init_backbone(),
        # we call it explicitly at each train epoch in case lightning calls
        # self.train() internally to change the states of dropout and batch norm
        if not self.hparams.finetune_tune_encoder:
            self.backbone.eval()
