#  classification on the learned representation.
#
# encoder:
# - set2set pooling
#
# decoders:
# - reaction type
#
from typing import Dict, List, Tuple

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
from rxnrep.model.readout import Set2SetThenCat


class LinearClassification(nn.Module):
    """
    Model to represent chemical reactions.
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

        # have reaction conv layer
        if reaction_conv_layer_sizes:
            conv_last_layer_size = reaction_conv_layer_sizes[-1]
        # does not have reaction conv layer
        else:
            conv_last_layer_size = molecule_conv_layer_sizes[-1]

        # readout reaction features, one 1D tensor for each reaction
        in_sizes = [conv_last_layer_size] * 2
        self.set2set = Set2SetThenCat(
            num_iters=set2set_num_iterations,
            num_layers=set2set_num_layers,
            ntypes=["atom", "bond"],
            in_feats=in_sizes,
            ntypes_direct_cat=["global"],
        )

        # linear classification head
        in_size = conv_last_layer_size * 5
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
