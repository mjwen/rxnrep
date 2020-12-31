from typing import Dict, Tuple

import dgl
import torch
import torch.nn as nn

from rxnrep.model.decoder import (
    AtomHopDistDecoder,
    AtomTypeDecoder,
    BondHopDistDecoder,
    FCNNDecoder,
    ReactionClusterDecoder,
    ReactionEnergyDecoder,
)
from rxnrep.model.encoder import ReactionEncoder
from rxnrep.model.readout import HopDistancePooling


class ReactionRepresentationHopDistPool(nn.Module):
    """
    Compared to ReactionRepresentation, change set2set pool to hop distance pool
    """

    def __init__(
        self,
        in_feats,
        embedding_size,
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
        # hop distance pooling
        max_hop_distance,
        # bond hop dist decoder
        bond_hop_dist_decoder_hidden_layer_sizes,
        bond_hop_dist_decoder_activation,
        bond_hop_dist_decoder_num_classes,
        # atom hop dist decoder
        atom_hop_dist_decoder_hidden_layer_sizes,
        atom_hop_dist_decoder_activation,
        atom_hop_dist_decoder_num_classes,
        # masked atom type decoder
        masked_atom_type_decoder_hidden_layer_sizes,
        masked_atom_type_decoder_activation,
        masked_atom_type_decoder_num_classes,
        # clustering decoder
        reaction_cluster_decoder_hidden_layer_sizes,
        reaction_cluster_decoder_activation,
        reaction_cluster_decoder_output_size,
    ):

        super(ReactionRepresentationHopDistPool, self).__init__()

        # encoder
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
            conv_last_layer_size = reaction_conv_layer_sizes[-1]
        # does not have reaction conv layer
        else:
            conv_last_layer_size = molecule_conv_layer_sizes[-1]

        # ========== node level decoder ==========

        # bond hop dist decoder
        self.bond_hop_dist_decoder = BondHopDistDecoder(
            in_size=conv_last_layer_size,
            num_classes=bond_hop_dist_decoder_num_classes,
            hidden_layer_sizes=bond_hop_dist_decoder_hidden_layer_sizes,
            activation=bond_hop_dist_decoder_activation,
        )

        # atom hop dist decoder
        self.atom_hop_dist_decoder = AtomHopDistDecoder(
            in_size=conv_last_layer_size,
            num_classes=atom_hop_dist_decoder_num_classes,
            hidden_layer_sizes=atom_hop_dist_decoder_hidden_layer_sizes,
            activation=atom_hop_dist_decoder_activation,
        )

        # masked atom type decoder
        self.masked_atom_type_decoder = AtomTypeDecoder(
            in_size=conv_last_layer_size,
            num_classes=masked_atom_type_decoder_num_classes,
            hidden_layer_sizes=masked_atom_type_decoder_hidden_layer_sizes,
            activation=masked_atom_type_decoder_activation,
        )

        # ========== reaction level decoder ==========

        # readout reaction features, one 1D tensor for each reaction
        self.hop_dist_pool = HopDistancePooling(max_hop=max_hop_distance)

        in_size = conv_last_layer_size * 3
        self.reaction_cluster_decoder = ReactionClusterDecoder(
            in_size=in_size,
            num_classes=reaction_cluster_decoder_output_size,
            hidden_layer_sizes=reaction_cluster_decoder_hidden_layer_sizes,
            activation=reaction_cluster_decoder_activation,
        )

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

        # readout reaction features, a 1D tensor for each reaction
        hop_dist = {
            "atom": metadata["atom_hop_dist"],
            "bond": metadata["bond_hop_dist"],
        }
        reaction_feats = self.hop_dist_pool(reaction_graphs, feats, hop_dist)

        return feats, reaction_feats

    def get_diff_feats(
        self,
        molecule_graphs: dgl.DGLGraph,
        reaction_graphs: dgl.DGLGraph,
        feats: Dict[str, torch.Tensor],
        metadata: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return self.encoder.get_diff_feats(
            molecule_graphs, reaction_graphs, feats, metadata
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

        # masked atom type decoder
        atom_ft = feats["atom"]
        masked_or_not = metadata["is_atom_masked"]
        atom_ft_of_masked_atoms = atom_ft[masked_or_not]
        masked_atom_type = self.masked_atom_type_decoder(atom_ft_of_masked_atoms)

        # reaction decoder
        reaction_cluster = self.reaction_cluster_decoder(reaction_feats)

        # predictions
        predictions = {
            "bond_hop_dist": bond_hop_dist,
            "atom_hop_dist": atom_hop_dist,
            "masked_atom_type": masked_atom_type,
            "reaction_cluster": reaction_cluster,
        }

        return predictions


class ReactionRepresentationHopDistPool2(ReactionRepresentationHopDistPool):
    """
    The same as ReactionRepresentationHopDistPool except that:
    1. a ReactionEnergyDecoder is used to train reaction features using reaction energy
    as labels.
    """

    def __init__(
        self,
        in_feats,
        embedding_size,
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
        # hop distance pooling
        max_hop_distance,
        # bond hop dist decoder
        bond_hop_dist_decoder_hidden_layer_sizes,
        bond_hop_dist_decoder_activation,
        bond_hop_dist_decoder_num_classes,
        # atom hop dist decoder
        atom_hop_dist_decoder_hidden_layer_sizes,
        atom_hop_dist_decoder_activation,
        atom_hop_dist_decoder_num_classes,
        # masked atom type decoder
        masked_atom_type_decoder_hidden_layer_sizes,
        masked_atom_type_decoder_activation,
        masked_atom_type_decoder_num_classes,
        # clustering decoder
        reaction_cluster_decoder_hidden_layer_sizes,
        reaction_cluster_decoder_activation,
        reaction_cluster_decoder_output_size,
        # reaction energy decoder
        reaction_energy_decoder_hidden_layer_sizes,
        reaction_energy_decoder_activation,
    ):

        super(ReactionRepresentationHopDistPool2, self).__init__(
            in_feats,
            embedding_size,
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
            # hop distance pooling
            max_hop_distance,
            # bond hop dist decoder
            bond_hop_dist_decoder_hidden_layer_sizes,
            bond_hop_dist_decoder_activation,
            bond_hop_dist_decoder_num_classes,
            # atom hop dist decoder
            atom_hop_dist_decoder_hidden_layer_sizes,
            atom_hop_dist_decoder_activation,
            atom_hop_dist_decoder_num_classes,
            # masked atom type decoder
            masked_atom_type_decoder_hidden_layer_sizes,
            masked_atom_type_decoder_activation,
            masked_atom_type_decoder_num_classes,
            # clustering decoder
            reaction_cluster_decoder_hidden_layer_sizes,
            reaction_cluster_decoder_activation,
            reaction_cluster_decoder_output_size,
        )

        # have reaction conv layer
        if reaction_conv_layer_sizes:
            conv_last_layer_size = reaction_conv_layer_sizes[-1]
        # does not have reaction conv layer
        else:
            conv_last_layer_size = molecule_conv_layer_sizes[-1]

        # reaction energy decoder
        in_size = conv_last_layer_size * 3
        self.reaction_energy_decoder = ReactionEnergyDecoder(
            in_size=in_size,
            hidden_layer_sizes=reaction_energy_decoder_hidden_layer_sizes,
            activation=reaction_energy_decoder_activation,
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

        # masked atom type decoder
        atom_ft = feats["atom"]
        masked_or_not = metadata["is_atom_masked"]
        atom_ft_of_masked_atoms = atom_ft[masked_or_not]
        masked_atom_type = self.masked_atom_type_decoder(atom_ft_of_masked_atoms)

        # reaction decoder
        reaction_cluster = self.reaction_cluster_decoder(reaction_feats)
        reaction_energy = self.reaction_energy_decoder(reaction_feats)

        # predictions
        predictions = {
            "bond_hop_dist": bond_hop_dist,
            "atom_hop_dist": atom_hop_dist,
            "masked_atom_type": masked_atom_type,
            "reaction_cluster": reaction_cluster,
            "reaction_energy": reaction_energy,
        }

        return predictions
