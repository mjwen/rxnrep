#
# A comprehensive model contains all decoders.
#
# encoder:
# - mol conv layers
# - diff mol feats
# - rxn conv layers (could be 0)
# - compression layers (could be 0)
# - pooling (set2set, hot distance)
#
# decoders:
# - atom hop dist
# - bond hop dist
# - masked atom hop
# - reaction clustering
#
# - reaction energy
# - activation energy
# - bep activation energy label
# - reaction type

# By default, the compressing layers and all decoders are set to None, meaning they
# will not be used. Also set the hidden_layer_sizes set an empty list will also not use
# the decoder.
#


from typing import Any, Dict

import torch

from rxnrep.model.encoder import ReactionEncoder
from rxnrep.model.utils import MLP


class ReactionRepresentation(ReactionEncoder):
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
        # compressing
        compressing_layer_sizes=None,
        compressing_layer_activation=None,
        # pooling
        pooling_method="set2set",
        pooling_kwargs: Dict[str, Any] = None,
        # bond hop dist decoder
        bond_hop_dist_decoder_hidden_layer_sizes=None,
        bond_hop_dist_decoder_activation=None,
        bond_hop_dist_decoder_num_classes=None,
        # atom hop dist decoder
        atom_hop_dist_decoder_hidden_layer_sizes=None,
        atom_hop_dist_decoder_activation=None,
        atom_hop_dist_decoder_num_classes=None,
        # masked atom type decoder
        masked_atom_type_decoder_hidden_layer_sizes=None,
        masked_atom_type_decoder_activation=None,
        masked_atom_type_decoder_num_classes=None,
        # reaction energy decoder
        reaction_energy_decoder_hidden_layer_sizes=None,
        reaction_energy_decoder_activation=None,
        # activation energy decoder
        activation_energy_decoder_hidden_layer_sizes=None,
        activation_energy_decoder_activation=None,
        # reaction classification decoder
        reaction_type_decoder_hidden_layer_sizes=None,
        reaction_type_decoder_num_classes=None,
        reaction_type_decoder_activation=None,
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
            compressing_layer_sizes=compressing_layer_sizes,
            compressing_layer_activation=compressing_layer_activation,
            pooling_method=pooling_method,
            pooling_kwargs=pooling_kwargs,
        )

        # ========== node level decoder ==========

        # bond hop dist decoder
        if bond_hop_dist_decoder_hidden_layer_sizes:
            self.bond_hop_dist_decoder = MLP(
                in_size=self.node_feats_size,
                hidden_sizes=bond_hop_dist_decoder_hidden_layer_sizes,
                activation=bond_hop_dist_decoder_activation,
                out_size=bond_hop_dist_decoder_num_classes,
            )
        else:
            self.bond_hop_dist_decoder = None

        # atom hop dist decoder
        if atom_hop_dist_decoder_hidden_layer_sizes:
            self.atom_hop_dist_decoder = MLP(
                in_size=self.node_feats_size,
                hidden_sizes=atom_hop_dist_decoder_hidden_layer_sizes,
                activation=atom_hop_dist_decoder_activation,
                out_size=atom_hop_dist_decoder_num_classes,
            )
        else:
            self.atom_hop_dist_decoder = None

        # masked atom type decoder
        if masked_atom_type_decoder_hidden_layer_sizes:
            self.masked_atom_type_decoder = MLP(
                in_size=self.node_feats_size,
                hidden_sizes=masked_atom_type_decoder_hidden_layer_sizes,
                activation=masked_atom_type_decoder_activation,
                out_size=masked_atom_type_decoder_num_classes,
            )
        else:
            self.masked_atom_type_decoder = None

        # ========== reaction level decoder ==========

        # reaction energy decoder
        if reaction_energy_decoder_hidden_layer_sizes:
            self.reaction_energy_decoder = MLP(
                in_size=self.reaction_feats_size,
                hidden_sizes=reaction_energy_decoder_hidden_layer_sizes,
                activation=reaction_energy_decoder_activation,
                out_size=1,
            )
        else:
            self.reaction_energy_decoder = None

        # activation energy decoder
        if activation_energy_decoder_hidden_layer_sizes:
            self.activation_energy_decoder = MLP(
                in_size=self.reaction_feats_size,
                hidden_sizes=activation_energy_decoder_hidden_layer_sizes,
                activation=activation_energy_decoder_activation,
                out_size=1,
            )
        else:
            self.activation_energy_decoder = None

        # reaction type decoder
        if reaction_type_decoder_hidden_layer_sizes:
            self.reaction_type_decoder = MLP(
                in_size=self.reaction_feats_size,
                hidden_sizes=reaction_type_decoder_hidden_layer_sizes,
                activation=reaction_type_decoder_activation,
                out_size=reaction_type_decoder_num_classes,
            )
        else:
            self.reaction_type_decoder = None

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
        bond_hop_dist = (
            None
            if self.bond_hop_dist_decoder is None
            else self.bond_hop_dist_decoder(feats["bond"])
        )

        # atom hop dist decoder
        atom_hop_dist = (
            None
            if self.atom_hop_dist_decoder is None
            else self.atom_hop_dist_decoder(feats["atom"])
        )

        # masked atom type decoder
        if self.masked_atom_type_decoder is None:
            masked_atom_type = None
        else:
            atom_ft = feats["atom"]
            masked_or_not = metadata["is_atom_masked"]
            atom_ft_of_masked_atoms = atom_ft[masked_or_not]
            masked_atom_type = self.masked_atom_type_decoder(atom_ft_of_masked_atoms)

        # reaction decoder
        reaction_energy = (
            None
            if self.reaction_energy_decoder is None
            else self.reaction_energy_decoder(reaction_feats)
        )

        activation_energy = (
            None
            if self.activation_energy_decoder is None
            else self.activation_energy_decoder(reaction_feats)
        )

        reaction_type = (
            None
            if self.reaction_type_decoder is None
            else self.reaction_type_decoder(reaction_feats)
        )

        # predictions
        predictions = {
            "bond_hop_dist": bond_hop_dist,
            "atom_hop_dist": atom_hop_dist,
            "masked_atom_type": masked_atom_type,
            "reaction_cluster": reaction_feats,
            "reaction_energy": reaction_energy,
            "activation_energy": activation_energy,
            "reaction_type": reaction_type,
        }

        return predictions
