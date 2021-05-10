"""
Adapt args in configs to create new ones for easier instantiation of the model
(encoder and decoder).
"""
from omegaconf import DictConfig, OmegaConf


def adjust_encoder_config(config: DictConfig) -> DictConfig:

    cfg = config.model.encoder

    new_config = OmegaConf.create({"model": {"encoder": {}}})
    new_cfg = new_config.model.encoder

    #
    # add new args
    #

    # has_global_feats: whether the conv model supports global features
    if cfg.conv in ["GINConv", "GINConvOriginal", "GATConv"]:
        new_cfg.has_global_feats = False
    elif cfg.conv in ["GINConvGlobal", "GatedGCNConv"]:
        new_cfg.has_global_feats = True
    else:
        raise ValueError(f"Unsupported conv {cfg.conv}")

    new_cfg.molecule_conv_layer_sizes = [cfg.conv_layer_size] * cfg.num_mol_conv_layers
    new_cfg.reaction_conv_layer_sizes = [cfg.conv_layer_size] * cfg.num_rxn_conv_layers

    # mlp after combining reactants and products feats
    new_cfg.mlp_diff_layer_sizes = [cfg.conv_layer_size] * cfg.num_mlp_diff_layers

    # mlp after pool
    size = determine_layer_size_by_pool_method(cfg)
    new_cfg.mlp_pool_layer_sizes = [size] * cfg.num_mlp_pool_layers

    #
    # adjust existing args
    #

    if not cfg.get("embedding_size", None):
        new_cfg.embedding_size = cfg.conv_layer_size

    # pool
    if not new_cfg.has_global_feats:
        new_cfg.pool_global_feats = False

    return new_config


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
