import torch.nn as nn
import torch
import dgl
from rxnrep.model.encoder import ReactionEncoder
from rxnrep.model.decoder import (
    BondTypeDecoder,
    AtomInReactionCenterDecoder,
    ReactionClusterDecoder,
)
from rxnrep.model.readout import Set2SetThenCat
from typing import List, Dict, Any, Tuple


class ReactionRepresentation(nn.Module):
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
        # bond type decoder
        bond_type_decoder_hidden_layer_sizes,
        bond_type_decoder_activation,
        # atom in reaction center decoder
        atom_in_reaction_center_decoder_hidden_layer_sizes,
        atom_in_reaction_center_decoder_activation,
        # clustering decoder
        reaction_cluster_decoder_hidden_layer_sizes,
        reaction_cluster_decoder_activation,
        reaction_cluster_decoder_output_size,
        # readout reaction features
        set2set_num_iterations: int = 3,
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

        # bond type decoder
        in_size = reaction_conv_layer_sizes[-1]
        self.bond_type_decoder = BondTypeDecoder(
            in_size=in_size,
            hidden_layer_sizes=bond_type_decoder_hidden_layer_sizes,
            activation=bond_type_decoder_activation,
        )

        # atom in reaction center decoder
        in_size = reaction_conv_layer_sizes[-1]
        self.atom_in_reaction_center_decoder = AtomInReactionCenterDecoder(
            in_size=in_size,
            hidden_layer_sizes=reaction_cluster_decoder_hidden_layer_sizes,
            activation=reaction_cluster_decoder_activation,
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
            hidden_layer_sizes=atom_in_reaction_center_decoder_hidden_layer_sizes,
            activation=atom_in_reaction_center_decoder_activation,
        )

    def forward(
        self,
        molecule_graphs: dgl.DGLGraph,
        reaction_graphs: dgl.DGLGraph,
        feats: Dict[str, torch.Tensor],
        metadata: Dict[str, List[int]],
    ) -> Tuple[Dict[str, Any], torch.Tensor]:
        """
        Args:
            molecule_graphs:
            reaction_graphs:
            feats:
            metadata:

        Returns:
            predictions: {decoder_name: value} predictions of the decoders.
            reaction_features: a tensor of the embeddings of all reactions.
        """
        # encoder
        feats = self.encoder(molecule_graphs, reaction_graphs, feats, metadata)

        ### node level decoder

        # bond type decoder
        bond_feats = feats["bond"]
        bond_type = self.bond_type_decoder(bond_feats)

        # atom in reaction center decoder
        atom_feats = feats["atom"]
        atom_in_reaction_center = self.atom_in_reaction_center_decoder(atom_feats)

        ### graph level decoder
        # readout reaction features, a 1D tensor for each reaction
        reaction_feats = self.set2set(reaction_graphs, feats)

        reaction_cluster = self.reaction_cluster_decoder(reaction_feats)

        ### predictions
        predictions = {
            "bond_type": bond_type,
            "atom_in_reaction_center": atom_in_reaction_center,
            "reaction_cluster": reaction_cluster,
        }

        return predictions, reaction_feats
