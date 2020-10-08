import torch.nn as nn
from typing import List, Callable


class UnifySize(nn.Module):
    """
    A layer to unify the feature size of nodes of different types.
    Each feature uses a linear fc layer to map the size.

    NOTE, after this transformation, each data point is just a linear combination of its
    feature in the original feature space (x_new_ij = x_ik w_kj), there is not mixing of
    feature between data points.

    Args:
        input_dim (dict): feature sizes of nodes with node type as key and size as value
        output_dim (int): output feature size, i.e. the size we will turn all the
            features to
    """

    def __init__(self, input_dim, output_dim):
        super(UnifySize, self).__init__()

        self.linears = nn.ModuleDict(
            {k: nn.Linear(size, output_dim, bias=False) for k, size in input_dim.items()}
        )

    def forward(self, feats):
        """
        Args:
            feats (dict): features dict with node type as key and feature as value

        Returns:
            dict: size adjusted features
        """
        return {k: self.linears[k](x) for k, x in feats.items()}


class FCNN(nn.Module):
    """
    A fully connected neural network.

    Args:
        in_size: input feature size
        out_sizes: size of each layer
        activations: activation function of each layer. If an element is `None`,
            then activation is not applied for that layer.
        use_bias: whether to use bias for each layer
    """

    def __init__(
        self,
        in_size: int,
        out_sizes: List[int],
        activations: List[Callable],
        use_bias: List[bool],
    ):
        super(FCNN, self).__init__()

        self.fc_layers = nn.ModuleList()
        for out, act, b in zip(out_sizes, activations, use_bias):
            self.fc_layers.append(nn.Linear(in_size, out, bias=b))
            if act is not None:
                self.fc_layers.append(act)
            in_size = out

    def forward(self, x):
        for layer in self.fc_layers:
            x = layer(x)
        return x

    def __repr__(self):
        return f"FCNN, num layers={len(self.fc_layers)}"
