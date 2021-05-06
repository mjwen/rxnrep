import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from rxnrep.model.encoder import ReactionEncoder
from rxnrep.model.losses import nt_xent_loss
from rxnrep.model.model_contrastive import BaseContrastiveModel
from rxnrep.model.utils import MLP
from rxnrep.utils.adapt_config import (
    adjust_encoder_config,
    determine_layer_size_by_pool_method,
)
from rxnrep.utils.config import merge_configs


def adjust_decoder_config(config: DictConfig) -> DictConfig:
    size = determine_layer_size_by_pool_method(config.model.encoder)
    minimum = config.model.encoder.conv_layer_size

    num_layers = config.model.decoder.model_class.simclr_decoder_num_layers
    layer_sizes = [max(size // 2 ** i, minimum) for i in range(num_layers)]

    new_config = OmegaConf.create(
        {
            "model": {
                "decoder": {"model_class": {"simclr_decoder_layer_sizes": layer_sizes}}
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


class LightningModel(BaseContrastiveModel):
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
        projection_decoder = MLP(
            in_size=self.backbone.reaction_feats_size,
            hidden_sizes=params.simclr_decoder_layer_sizes[:-1],
            activation=params.activation,
            out_size=params.simclr_decoder_layer_sizes[-1],
        )

        return {"projection_head_decoder": projection_decoder}

    def decode(self, feats, reaction_feats, metadata):
        decoder = self.decoder["projection_head_decoder"]
        z = decoder(reaction_feats)

        return z

    def compute_loss(self, preds, labels):
        z1 = preds["z1"]
        z2 = preds["z2"]

        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        loss = nt_xent_loss(z1, z2, self.hparams.simclr_temperature)

        return {"contrastive": loss}
