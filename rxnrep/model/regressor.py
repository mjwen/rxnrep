import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from rxnrep.layer.encoder import ReactionEncoder, adjust_encoder_config
from rxnrep.layer.utils import MLP
from rxnrep.model.base_model import BaseModel
from rxnrep.utils.config import determine_layer_size_by_pool_method, merge_configs


def adjust_decoder_config(config: DictConfig):
    size = determine_layer_size_by_pool_method(config.model.encoder)

    num_layers = config.model.decoder.model_class.regression_decoder_num_layers

    # this is for multi property regression
    hidden_layer_sizes = []
    for n in num_layers:
        hidden_layer_sizes.append([max(size // 2 ** i, 50) for i in range(n)])

    new_config = OmegaConf.create(
        {
            "model": {
                "decoder": {
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
    Adjust model config, both encoder and decoder.
    """
    encoder_config = adjust_encoder_config(config)

    # create a temporary one to merge original and encoder
    # this is needed since info from both config is needed to adjust decoder config
    merged = merge_configs(config, encoder_config)
    decoder_config = adjust_decoder_config(merged)
    model_config = merge_configs(encoder_config, decoder_config)

    return model_config


class LightningModel(BaseModel):
    def init_backbone(self, params):
        model = ReactionEncoder(
            in_feats=params.dataset_info["feature_size"],
            embedding_size=params.embedding_size,
            # encoder
            molecule_conv_layer_sizes=params.molecule_conv_layer_sizes,
            molecule_num_fc_layers=params.molecule_num_fc_layers,
            molecule_batch_norm=params.molecule_batch_norm,
            molecule_activation=params.activation,
            molecule_residual=params.molecule_residual,
            molecule_dropout=params.molecule_dropout,
            conv=params.conv,
            combine_reactants_products=params.combine_reactants_products,
        )

        return model

    def init_decoder(self, params):
        decoder = {}

        for name, size in zip(
            params.property_name, params.regression_decoder_hidden_layer_sizes
        ):
            name = name + "_decoder"  # e.g. reaction_energy_decoder

            decoder[name] = MLP(
                in_size=self.backbone.reaction_feats_size,
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
