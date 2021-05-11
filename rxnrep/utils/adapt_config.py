"""
Adapt args in configs to create new ones for easier instantiation of the model
(encoder and decoder).
"""


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
