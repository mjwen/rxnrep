import torch
import torch.nn as nn
from rxnrep.model.utils import FCNN
from typing import List


class NodeClassificationDecoder(nn.Module):
    """
    A base class implementing classification based on node features (either atom or
    bond node).

    Args:
        in_size: input size of the features
        hidden_layer_sizes: size of the hidden layers to transform the features.
            Note, there will be an additional layer applied after this,
            which transforms the features to `num_classes` dimensions.
        activation: activation function applied after the hidden layer
        num_classes: number of classes

    """

    def __init__(
        self,
        in_size: int,
        hidden_layer_sizes: List[int] = None,
        activation: str = "ReLU",
        num_classes: int = 3,
    ):
        super(NodeClassificationDecoder, self).__init__()

        # set default values
        if hidden_layer_sizes is None:
            hidden_layer_sizes = [64, 32]
        if isinstance(activation, str):
            activation = getattr(nn, activation)()

        # no activation for last layer
        out_sizes = hidden_layer_sizes + [num_classes]
        use_bias = [True] * len(out_sizes)
        acts = [activation] * len(hidden_layer_sizes) + [None]

        self.fc_layers = FCNN(in_size, out_sizes, acts, use_bias)

    def forward(self, feats: torch.Tensor):
        """
        Args:
            feats: bond features, a 2D tensor of shape (N, D).

        Returns:
            Updated features, a 2D tensor of shape (N, num_classes),
            where `num_classes` is the number of classes.
        """

        return self.fc_layers(feats)


class BondTypeDecoder(NodeClassificationDecoder):
    """
    A decoder to predict the bond type from bond features.

    The bond features are first `decoded` via a FCNN, and then passed through a via a
    cross entropy loss.

    There are three types of bond:
    1. unchanged bond: bonds exists in both the reactants and the products
    2. lost bond: bonds in the reactants breaks in a reaction
    3. added bonds: bonds in the products created in a reaction

    Args:
        in_size: input size of the bond features
        hidden_layer_sizes: size of the hidden layers to transform the bond features.
            Note, there will be an additional layer applied after this,
            which transforms the features to `num_classes` dimensions.
        activation: activation function applied after the hidden layer
        num_classes: number of bond type classes.
    """

    def __init__(
        self,
        in_size: int,
        hidden_layer_sizes: List[int] = None,
        activation: str = "ReLU",
        num_classes: int = 3,
    ):
        super(BondTypeDecoder, self).__init__(
            in_size, hidden_layer_sizes, activation, num_classes
        )


class AtomInReactionCenterDecoder(NodeClassificationDecoder):
    """
    A decoder to predict whether an atom is in the reaction center from the atom features.

    The atom features are first `decoded` via a FCNN, and then passed through a binary
    cross entry loss.

    Args:
        in_size: input size of the atom features
        hidden_layer_sizes: size of the hidden layers to transform the atom features.
            Note, there will be an additional layer applied after this,
            which transforms the features to a (N 1) tensor.
        activation: activation function applied after the hidden layer
        """

    def __init__(
        self,
        in_size: int,
        hidden_layer_sizes: List[int] = None,
        activation: str = "ReLU",
    ):
        # Note, for binary classification, the sigmoid function takes a scalar,
        # so `num_classes` is set to 1
        super(AtomInReactionCenterDecoder, self).__init__(
            in_size, hidden_layer_sizes, activation, num_classes=1
        )
