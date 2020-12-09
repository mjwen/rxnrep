from typing import Any, Dict, List, Tuple

import dgl
import torch
import torch.nn as nn

from rxnrep.model.decoder import (
    AtomHopDistDecoder,
    AtomTypeDecoder,
    BondHopDistDecoder,
    FCNNDecoder,
    ReactionClusterDecoder,
)
from rxnrep.model.encoder import ReactionEncoder
from rxnrep.model.readout import Set2SetThenCat


class ReactionRepresentation(nn.Module):
    """
    Model to represent chemical reactions.


    Args:
        in_feats:
        embedding_size:
        molecule_conv_layer_sizes:
        molecule_num_fc_layers:
        molecule_batch_norm:
        molecule_activation:
        molecule_residual:
        molecule_dropout:
        reaction_conv_layer_sizes:
        reaction_num_fc_layers:
        reaction_batch_norm:
        reaction_activation:
        reaction_residual:
        reaction_dropout:
        bond_hop_dist_decoder_hidden_layer_sizes:
        bond_hop_dist_decoder_activation:
        bond_hop_dist_decoder_num_classes:
        atom_hop_dist_decoder_hidden_layer_sizes:
        atom_hop_dist_decoder_activation:
        atom_hop_dist_decoder_num_classes:
        masked_atom_type_decoder_hidden_layer_sizes,
        masked_atom_type_decoder_activation,
        masked_atom_type_decoder_num_classes,
        reaction_cluster_decoder_hidden_layer_sizes:
        reaction_cluster_decoder_activation:
        reaction_cluster_decoder_output_size:
        set2set_num_iterations:
        set2set_num_layers:
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
        # readout reaction features
        set2set_num_iterations: int = 6,
        set2set_num_layers: int = 3,
    ):

        super(ReactionRepresentation, self).__init__()

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

        # ========== node level decoder ==========

        # bond hop dist decoder
        in_size = reaction_conv_layer_sizes[-1]
        self.bond_hop_dist_decoder = BondHopDistDecoder(
            in_size=in_size,
            num_classes=bond_hop_dist_decoder_num_classes,
            hidden_layer_sizes=bond_hop_dist_decoder_hidden_layer_sizes,
            activation=bond_hop_dist_decoder_activation,
        )

        # atom hop dist decoder
        in_size = reaction_conv_layer_sizes[-1]
        self.atom_hop_dist_decoder = AtomHopDistDecoder(
            in_size=in_size,
            num_classes=atom_hop_dist_decoder_num_classes,
            hidden_layer_sizes=atom_hop_dist_decoder_hidden_layer_sizes,
            activation=atom_hop_dist_decoder_activation,
        )

        # masked atom type decoder
        in_size = reaction_conv_layer_sizes[-1]
        self.masked_atom_type_decoder = AtomTypeDecoder(
            in_size=in_size,
            num_classes=masked_atom_type_decoder_num_classes,
            hidden_layer_sizes=masked_atom_type_decoder_hidden_layer_sizes,
            activation=masked_atom_type_decoder_activation,
        )

        # ========== reaction level decoder ==========

        # readout reaction features, one 1D tensor for each reaction
        in_sizes = [reaction_conv_layer_sizes[-1]] * 2
        self.set2set = Set2SetThenCat(
            num_iters=set2set_num_iterations,
            num_layers=set2set_num_layers,
            ntypes=["atom", "bond"],
            in_feats=in_sizes,
            ntypes_direct_cat=["global"],
        )

        in_size = reaction_conv_layer_sizes[-1] * 5
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
    ) -> Tuple[Dict[str, Any], torch.Tensor]:
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
        reaction_feats = self.set2set(reaction_graphs, feats)

        return feats, reaction_feats

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


class LinearClassification(nn.Module):
    """
    Model to represent chemical reactions.


    Args:
        in_feats:
        embedding_size:
        molecule_conv_layer_sizes:
        molecule_num_fc_layers:
        molecule_batch_norm:
        molecule_activation:
        molecule_residual:
        molecule_dropout:
        reaction_conv_layer_sizes:
        reaction_num_fc_layers:
        reaction_batch_norm:
        reaction_activation:
        reaction_residual:
        reaction_dropout:
        num_classes: number of reaction classes
        set2set_num_iterations:
        set2set_num_layers:
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
        # classification head
        head_hidden_layer_sizes,
        num_classes,
        head_activation,
        # readout reaction features
        set2set_num_iterations: int = 6,
        set2set_num_layers: int = 3,
    ):

        super(LinearClassification, self).__init__()

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

        # readout reaction features, one 1D tensor for each reaction
        in_sizes = [reaction_conv_layer_sizes[-1]] * 2
        self.set2set = Set2SetThenCat(
            num_iters=set2set_num_iterations,
            num_layers=set2set_num_layers,
            ntypes=["atom", "bond"],
            in_feats=in_sizes,
            ntypes_direct_cat=["global"],
        )

        # linear classification head
        in_size = reaction_conv_layer_sizes[-1] * 5
        self.classification_head = FCNNDecoder(
            in_size, num_classes, head_hidden_layer_sizes, head_activation
        )

    def forward(
        self,
        molecule_graphs: dgl.DGLGraph,
        reaction_graphs: dgl.DGLGraph,
        feats: Dict[str, torch.Tensor],
        metadata: Dict[str, List[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        reaction_feats = self.set2set(reaction_graphs, feats)

        return feats, reaction_feats

    def decode(self, feats: torch.Tensor, reaction_feats: torch.Tensor) -> torch.Tensor:
        """
        Decode the molecule and reaction features to properties.

        Args:
            feats: first output of `forward()`
            reaction_feats: second output of `forward()`

        Returns:
            logits: a tensor of shape (N, num_classes), where N is the number of data
                points, and num_classes is the number of reaction classes
        """

        # classification head
        logits = self.classification_head(reaction_feats)

        return logits
