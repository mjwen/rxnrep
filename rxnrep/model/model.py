import torch.nn as nn
import torch
import dgl
from rxnrep.model.encoder import ReactionEncoder
from rxnrep.model.decoder import BondTypeDecoder, AtomInReactionCenterDecoder
from typing import List, Dict, Any


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
            hidden_layer_sizes=atom_in_reaction_center_decoder_hidden_layer_sizes,
            activation=atom_in_reaction_center_decoder_activation,
        )

    def forward(
        self,
        molecule_graphs: dgl.DGLGraph,
        reaction_graphs: dgl.DGLGraph,
        feats: Dict[str, torch.Tensor],
        metadata: Dict[str, List[int]],
    ) -> Dict[str, Any]:
        """
        Args:
            molecule_graphs:
            reaction_graphs:
            feats:
            metadata:

        Returns:
            {decoder_name: value}: predictions of the decoders.

        """
        # encoder
        feats = self.encoder(molecule_graphs, reaction_graphs, feats, metadata)

        # bond type decoder
        bond_feats = feats["bond"]
        bond_type = self.bond_type_decoder(bond_feats)

        # atom in reaction center decoder
        atom_feats = feats["atom"]
        atom_in_reaction_center = self.atom_in_reaction_center_decoder(atom_feats)

        predictions = {
            "bond_type": bond_type,
            "atom_in_reaction_center": atom_in_reaction_center,
        }

        return predictions
