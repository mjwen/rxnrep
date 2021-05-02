"""
This file holds some utility function to convert hydra config arguments to arguments
for the model.
"""

from omegaconf import DictConfig


def adjust_encoder_config(config: DictConfig):

    cfg = config.model.encoder

    #
    # add new args
    #

    # has_global_feats: whether the conv model supports global features
    if cfg.conv in ["GINConv", "GINConvOriginal"]:
        cfg.has_global_feats = False
    elif cfg.conv in ["GINConvGlobal", "GatedGCNConv"]:
        cfg.has_global_feats = True
    else:
        raise ValueError(f"Unsupported conv {cfg.encoder.conv}")

    cfg.molecule_conv_layer_sizes = [cfg.conv_layer_size] * cfg.num_mol_conv_layers
    cfg.reaction_conv_layer_sizes = [cfg.conv_layer_size] * cfg.num_rxn_conv_layers

    # mlp after combining reactants and products feats
    cfg.mlp_diff_layer_sizes = [cfg.conv_layer_size] * cfg.num_mlp_diff_layers

    # mlp after pool
    size = determine_layer_size_by_pool_method(cfg)
    cfg.mlp_pool_layer_sizes = [size] * cfg.num_mlp_pool_layers

    #
    # adjust existing args
    #

    if not cfg.get("embedding_size", None):
        cfg.embedding_size = cfg.conv_layer_size

    # pool
    if not cfg.has_global_feats:
        cfg.pool_global_feats = False


def adjust_reaction_type_decoder(config: DictConfig):
    size = determine_layer_size_by_pool_method(config.model.encoder)

    #
    # add new args
    #
    cfg = config.model.decoder.model_class
    cfg.reaction_type_decoder_hidden_layer_sizes = [
        max(size // 2 ** i, 50) for i in range(cfg.reaction_type_decoder_num_layers)
    ]


def determine_layer_size_by_pool_method(encoder_cfg):
    n = sum(
        [
            encoder_cfg.pool_atom_feats,
            encoder_cfg.pool_bond_feats,
            encoder_cfg.pool_global_feats,
        ]
    )
    size = n * encoder_cfg.conv_layer_size

    return size
