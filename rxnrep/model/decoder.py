import torch
import torch.nn as nn
from rxnrep.model.utils import FCNN
from typing import Dict, List


def create_label_bond_type_decoder(metadata: Dict[str, List[int]]) -> torch.Tensor:
    """
    Unchanged bond with class 0, lost bond with class 1, and added bond with class 2.

    Returns:
        1D tensor of the class for each bond.
    """
    num_unchanged_bonds = metadata["num_unchanged_bonds"]
    num_lost_bonds = metadata["num_lost_bonds"]
    num_added_bonds = metadata["num_added_bonds"]

    labels = []
    for unchanged, lost, added in zip(
        num_unchanged_bonds, num_lost_bonds, num_added_bonds
    ):
        labels.extend([0] * unchanged + [1] * lost + [2] * added)

    labels = torch.tensor(labels, dtype=torch.int64)

    return labels


class BondTypeDecoder(nn.Module):
    """
    A decoder to predict the Bond type between two atoms.

    The takes the reaction graph and predicts the bond type from the bond features.
    The bond features are first `decoded` via a FCNN


    There are three types of bond:
    1. unchanged bond: bonds exists in both the reactants and the products
    2. lost bond: bonds in the reactants breaks in a reaction
    3. added bonds: bonds in the products created in a reaction

    Args:
        in_size: input size of the bond features
        hidden_layer_sizes: size of the hidden layers to transform the bond
            features to logits. Note, there will be an additional layer applied after
            this, which will have a size of `num_classes`.
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
        super(BondTypeDecoder, self).__init__()

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
            logits of each bond feature, a 2D tensor of shape (N, num_classes),
            where num_classes is the number of bond classes.
        """

        return self.fc_layers(feats)
