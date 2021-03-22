from typing import Callable, List, Optional, Union

import torch.nn as nn


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
            {
                k: nn.Linear(size, output_dim, bias=False)
                for k, size in input_dim.items()
            }
        )

    def forward(self, feats):
        """
        Args:
            feats (dict): features dict with node type as key and feature as value

        Returns:
            dict: size adjusted features
        """
        return {k: self.linears[k](x) for k, x in feats.items()}


class MLP(nn.Module):
    """
    Multilayer perceptrons.

    By default, activation is applied to each hidden layer. Optionally, one can asking
    for an output layer by setting out_size.


    Args:
        in_size: input feature size
        hidden_sizes: sizes for hidden layers
        batch_norm: whether to add 1D batch norm
        activation: activation function for hidden layers
        out_size: size of output layer
        out_batch_norm: whether to add 1D batch norm for output layer
        out_activation: whether to apply activation for output layer
    """

    def __init__(
        self,
        in_size: int,
        hidden_sizes: List[int],
        *,
        batch_norm: bool = False,
        activation: Union[Callable, str] = "ReLU",
        out_size: Optional[int] = None,
        out_batch_norm: bool = False,
        out_activation: bool = False,
    ):
        super(MLP, self).__init__()
        self.num_hidden_layers = len(hidden_sizes)
        self.has_out_layer = out_size is not None

        layers = []

        # hidden layers
        if batch_norm:
            bias = False
        else:
            bias = True

        for size in hidden_sizes:
            layers.append(nn.Linear(in_size, size, bias=bias))

            if batch_norm:
                layers.append(nn.BatchNorm1d(size))

            if activation is not None:
                layers.append(get_activation(activation))

            in_size = size

        # output layer
        if out_batch_norm:
            out_bias = False
        else:
            out_bias = True

        if out_size is not None:
            layers.append(nn.Linear(in_size, out_size, bias=out_bias))

            if out_batch_norm:
                layers.append(nn.BatchNorm1d(out_size))

            if activation is not None and out_activation:
                layers.append(get_activation(activation))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

    def __repr__(self):
        s = f"MLP, num hidden layers: {self.num_layers}"
        if self.has_out_layer:
            s += "; with output layer"
        return s


def get_activation(act: Union[str, Callable]) -> Callable:
    """
    Get the activation function.

    If it is a string, convert to torch activation function; if it is already a torch
    activation function, simply return it.
    """
    if isinstance(act, str):
        act = getattr(nn, act)()
    return act


def get_dropout(drop_ratio: float, delta=1e-3) -> Callable:
    """
    Get dropout and do not use it if ratio is smaller than delta.
    """

    if drop_ratio is None or drop_ratio < delta:
        return nn.Identity()
    else:
        return nn.Dropout(drop_ratio)
