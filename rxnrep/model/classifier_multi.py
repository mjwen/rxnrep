import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from rxnrep.model.encoder import ReactionEncoder
from rxnrep.model.model import BaseModel
from rxnrep.model.utils import MLP
from rxnrep.utils.adjust_config import (
    adjust_encoder_config,
    determine_layer_size_by_pool_method,
)


def adjust_decoder_config(config: DictConfig):
    size = determine_layer_size_by_pool_method(config.model.encoder)

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


def adjust_config(config: DictConfig) -> DictConfig:
    encoder_config = adjust_encoder_config(config)

    # create a temporary one to merge original and encoder
    OmegaConf.set_struct(config, False)
    merged = OmegaConf.merge(config, encoder_config)
    OmegaConf.set_struct(config, True)

    decoder_config = adjust_decoder_config(merged)

    return OmegaConf.merge(encoder_config, decoder_config)


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
            reaction_conv_layer_sizes=params.reaction_conv_layer_sizes,
            reaction_num_fc_layers=params.reaction_num_fc_layers,
            reaction_batch_norm=params.reaction_batch_norm,
            reaction_activation=params.activation,
            reaction_residual=params.reaction_residual,
            reaction_dropout=params.reaction_dropout,
            #
            conv=params.conv,
            has_global_feats=params.has_global_feats,
            #
            combine_reactants_products=params.combine_reactants_products,
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
        )

        return model

    def init_decoder(self, params):
        decoder = MLP(
            in_size=self.backbone.reaction_feats_size,
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
