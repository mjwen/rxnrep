#
# Basic representational learning model.
# Others should build by modifying this code, not to subclass for easier comparison.
#
# encoder:
# - mol conv layers
# - diff mol feats
# - rxn conv layers (could be 0)
# - pooling (set2set, hot distance)
# - compression layers (could be 0)
#
# decoders:
# - atom hop dist
# - bond hop dist
# - masked atom hop
# - reaction clustering
#
from typing import Any, Dict, Tuple

import dgl
import torch
import torch.nn as nn

from rxnrep.model.decoder import (
    AtomHopDistDecoder,
    AtomTypeDecoder,
    BondHopDistDecoder,
    ReactionClusterDecoder,
)
from rxnrep.model.encoder import ReactionEncoder
from rxnrep.model.readout import CompressingNN, HopDistancePooling, Set2SetThenCat


class EncoderAndPooling(nn.Module):
    """
    Encoder and reaction feature pooling part of the model. Add decoder to use this.
    """

    def __init__(
        self,
        in_feats,
        embedding_size,
        *,
        # encoder
        molecule_conv_layer_sizes,
        molecule_num_fc_layers,
        molecule_batch_norm,
        molecule_activation,
        molecule_residual,
        molecule_dropout,
        reaction_conv_layer_sizes,
        reaction_num_fc_layers,
        reaction_batch_norm,
        reaction_activation,
        reaction_residual,
        reaction_dropout,
        # compressing
        compressing_layer_sizes=None,
        compressing_layer_activation=None,
        # pooling
        pooling_method="set2set",
        pooling_kwargs: Dict[str, Any] = None,
    ):

        super().__init__()

        # ========== encoder ==========
        self.encoder = ReactionEncoder(
            in_feats=in_feats,
            embedding_size=embedding_size,
            molecule_conv_layer_sizes=molecule_conv_layer_sizes,
            molecule_num_fc_layers=molecule_num_fc_layers,
            molecule_batch_norm=molecule_batch_norm,
            molecule_activation=molecule_activation,
            molecule_residual=molecule_residual,
            molecule_dropout=molecule_dropout,
            reaction_conv_layer_sizes=reaction_conv_layer_sizes,
            reaction_num_fc_layers=reaction_num_fc_layers,
            reaction_batch_norm=reaction_batch_norm,
            reaction_activation=reaction_activation,
            reaction_residual=reaction_residual,
            reaction_dropout=reaction_dropout,
        )

        # have reaction conv layer
        if reaction_conv_layer_sizes:
            encoder_outsize = reaction_conv_layer_sizes[-1]
        # does not have reaction conv layer
        else:
            encoder_outsize = molecule_conv_layer_sizes[-1]

        # ========== compressor ==========
        if compressing_layer_sizes:
            self.compressor = nn.ModuleDict(
                {
                    k: CompressingNN(
                        in_size=encoder_outsize,
                        hidden_sizes=compressing_layer_sizes,
                        activation=compressing_layer_activation,
                    )
                    for k in ["atom", "bond", "global"]
                }
            )
            compressor_outsize = compressing_layer_sizes[-1]
        else:
            self.compressor = nn.ModuleDict(
                {k: nn.Identity() for k in ["atom", "bond", "global"]}
            )
            compressor_outsize = encoder_outsize

        # ========== reaction feature pooling ==========
        # readout reaction features, one 1D tensor for each reaction

        self.pooling_method = pooling_method

        if pooling_method == "set2set":
            if pooling_kwargs is None:
                set2set_num_iterations = 6
                set2set_num_layers = 3
            else:
                set2set_num_iterations = pooling_kwargs["set2set_num_iterations"]
                set2set_num_layers = pooling_kwargs["set2set_num_layers"]

            in_sizes = [compressor_outsize] * 2
            self.set2set = Set2SetThenCat(
                num_iters=set2set_num_iterations,
                num_layers=set2set_num_layers,
                ntypes=["atom", "bond"],
                in_feats=in_sizes,
                ntypes_direct_cat=["global"],
            )

            pooling_outsize = compressor_outsize * 5

        elif pooling_method == "hop_distance":
            if pooling_kwargs is None:
                raise RuntimeError(
                    "`max_hop_distance` should be provided as `pooling_kwargs` to use "
                    "`hop_distance_pool`"
                )
            else:
                max_hop_distance = pooling_kwargs["max_hop_distance"]
                self.hop_dist_pool = HopDistancePooling(max_hop=max_hop_distance)

            pooling_outsize = compressor_outsize * 3

        else:
            raise ValueError(f"Unsupported pooling method `{pooling_method}`")

        self.node_feats_size = compressor_outsize
        self.reaction_feats_size = pooling_outsize

    def forward(
        self,
        molecule_graphs: dgl.DGLGraph,
        reaction_graphs: dgl.DGLGraph,
        feats: Dict[str, torch.Tensor],
        metadata: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        We let forward only returns features and the part to map the features to logits
        in another function: `decode`.

        Args:
            molecule_graphs:
            reaction_graphs:
            feats:
            metadata:
        """
        # encoder
        feats = self.encoder(molecule_graphs, reaction_graphs, feats, metadata)

        # compressor
        feats = {k: self.compressor[k](feats[k]) for k in ["atom", "bond", "global"]}

        # readout reaction features, a 1D tensor for each reaction
        if self.pooling_method == "set2set":
            reaction_feats = self.set2set(reaction_graphs, feats)

        elif self.pooling_method == "hop_distance":

            hop_dist = {
                "atom": metadata["atom_hop_dist"],
                "bond": metadata["bond_hop_dist"],
            }
            reaction_feats = self.hop_dist_pool(reaction_graphs, feats, hop_dist)

        else:
            raise ValueError(f"Unsupported pooling method `{self.pooling_method}`")

        return feats, reaction_feats

    def get_diff_feats(
        self,
        molecule_graphs: dgl.DGLGraph,
        reaction_graphs: dgl.DGLGraph,
        feats: Dict[str, torch.Tensor],
        metadata: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Get the atom/bond/global difference features before applying reaction
        convolution.

        Returns:
            {atom:feats, bond:feats, global:feats}

        """

        return self.encoder.get_diff_feats(
            molecule_graphs, reaction_graphs, feats, metadata
        )
