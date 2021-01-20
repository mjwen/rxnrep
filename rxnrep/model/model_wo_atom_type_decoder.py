#
# decoders:
# - atom hop dist
# - bond hop dist
# - reaction clustering
#
from typing import Any, Dict

import torch

from rxnrep.model.decoder import (
    AtomHopDistDecoder,
    BondHopDistDecoder,
    ReactionClusterDecoder,
)
from rxnrep.model.model import EncoderAndPooling


class ReactionRepresentation(EncoderAndPooling):
    """
    Model to represent chemical reactions.
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
        # bond hop dist decoder
        bond_hop_dist_decoder_hidden_layer_sizes,
        bond_hop_dist_decoder_activation,
        bond_hop_dist_decoder_num_classes,
        # atom hop dist decoder
        atom_hop_dist_decoder_hidden_layer_sizes,
        atom_hop_dist_decoder_activation,
        atom_hop_dist_decoder_num_classes,
        # masked atom type decoder
        # masked_atom_type_decoder_hidden_layer_sizes,
        # masked_atom_type_decoder_activation,
        # masked_atom_type_decoder_num_classes,
        # clustering decoder
        reaction_cluster_decoder_hidden_layer_sizes,
        reaction_cluster_decoder_activation,
        reaction_cluster_decoder_output_size,
        # reaction features pooling
        pooling_method="set2set",
        pooling_kwargs: Dict[str, Any] = None,
    ):

        # encoder and pooling
        super().__init__(
            in_feats,
            embedding_size,
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
            pooling_method=pooling_method,
            pooling_kwargs=pooling_kwargs,
        )

        # ========== node level decoder ==========

        # bond hop dist decoder
        self.bond_hop_dist_decoder = BondHopDistDecoder(
            in_size=self.conv_last_layer_size,
            num_classes=bond_hop_dist_decoder_num_classes,
            hidden_layer_sizes=bond_hop_dist_decoder_hidden_layer_sizes,
            activation=bond_hop_dist_decoder_activation,
        )

        # atom hop dist decoder
        self.atom_hop_dist_decoder = AtomHopDistDecoder(
            in_size=self.conv_last_layer_size,
            num_classes=atom_hop_dist_decoder_num_classes,
            hidden_layer_sizes=atom_hop_dist_decoder_hidden_layer_sizes,
            activation=atom_hop_dist_decoder_activation,
        )

        # # masked atom type decoder
        # self.masked_atom_type_decoder = AtomTypeDecoder(
        #     in_size=self.conv_last_layer_size,
        #     num_classes=masked_atom_type_decoder_num_classes,
        #     hidden_layer_sizes=masked_atom_type_decoder_hidden_layer_sizes,
        #     activation=masked_atom_type_decoder_activation,
        # )

        # ========== reaction level decoder ==========

        self.reaction_cluster_decoder = ReactionClusterDecoder(
            in_size=self.pooling_last_layer_size,
            num_classes=reaction_cluster_decoder_output_size,
            hidden_layer_sizes=reaction_cluster_decoder_hidden_layer_sizes,
            activation=reaction_cluster_decoder_activation,
        )

    def decode(
        self,
        feats: torch.Tensor,
        reaction_feats: torch.Tensor,
        metadata: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Decode the molecule and reaction features to properties.

        Args:
            feats: first output of `forward()`
            reaction_feats: second output of `forward()`
            metadata:

        Returns:
            predictions: {decoder_name: value} predictions of the decoders.
        """

        # bond hop dist decoder
        bond_hop_dist = self.bond_hop_dist_decoder(feats["bond"])

        # atom hop dist decoder
        atom_hop_dist = self.atom_hop_dist_decoder(feats["atom"])

        # # masked atom type decoder
        # atom_ft = feats["atom"]
        # masked_or_not = metadata["is_atom_masked"]
        # atom_ft_of_masked_atoms = atom_ft[masked_or_not]
        # masked_atom_type = self.masked_atom_type_decoder(atom_ft_of_masked_atoms)

        # reaction decoder
        reaction_cluster = self.reaction_cluster_decoder(reaction_feats)

        # predictions
        predictions = {
            "bond_hop_dist": bond_hop_dist,
            "atom_hop_dist": atom_hop_dist,
            # "masked_atom_type": masked_atom_type,
            "reaction_cluster": reaction_cluster,
        }

        return predictions