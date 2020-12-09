from typing import List

import torch
import torch.nn as nn

from rxnrep.model.utils import FCNN


class BaseDecoder(nn.Module):
    """
    A base decoder class to map the features before passing them to sigmoid or softmax
    layer to compute logits.

    This uses N fully connected layer as the decoder.

    Note, there will be an additional layer applied after the hidden layers,
    which transforms the features to `num_classes` dimensions without applying the
    activation.

    Args:
        in_size: input size of the features
        num_classes: number of classes
        hidden_layer_sizes: size of the hidden layers to transform the features.
            Note, there will be an additional layer applied after this,
            which transforms the features to `num_classes` dimensions.
        activation: activation function applied after the hidden layer
    """

    def __init__(
        self,
        in_size: int,
        num_classes: int,
        hidden_layer_sizes: List[int],
        activation: str = "ReLU",
    ):
        super(BaseDecoder, self).__init__()

        # no activation for last layer
        out_sizes = hidden_layer_sizes + [num_classes]
        use_bias = [True] * len(out_sizes)
        acts = [activation] * len(hidden_layer_sizes) + [None]

        self.fc_layers = FCNN(in_size, out_sizes, acts, use_bias)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feats: features, a 2D tensor of shape (N, D), where N is the batch size and
                D is the feature dimension.

        Returns:
            Updated features, a 2D tensor of shape (N, num_classes), where `num_classes`
            is the number of classes.
        """

        return self.fc_layers(feats)


class LinearClassificationHead(nn.Module):
    """
    A linear layer, y=Wx+b, to project the learned features to class labels.

    This can be used together with a cross entropy loss to evaluate classification error.

    Args:
        in_size: input size of the features
        num_classes: number of classes
    """

    def __init__(self, in_size: int, num_classes: int, use_bias=True):
        super(LinearClassificationHead, self).__init__()
        self.layer = nn.Linear(in_size, num_classes, use_bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layer(features)


BondHopDistDecoder = BaseDecoder
AtomHopDistDecoder = BaseDecoder
ReactionClusterDecoder = BaseDecoder
AtomTypeDecoder = BaseDecoder
FCNNDecoder = BaseDecoder
