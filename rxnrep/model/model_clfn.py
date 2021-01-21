#
# decoders:
# - reaction type
#
from typing import Any, Dict

import torch

from rxnrep.model.decoder import FCNNDecoder
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
        # classification head
        head_hidden_layer_sizes,
        num_classes,
        head_activation,
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

        # linear classification head
        self.classification_head = FCNNDecoder(
            self.reaction_feats_size,
            num_classes,
            head_hidden_layer_sizes,
            head_activation,
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
            logits: a tensor of shape (N, num_classes), where N is the number of data
                points, and num_classes is the number of reaction classes
        """

        # classification head
        logits = self.classification_head(reaction_feats)

        return logits
