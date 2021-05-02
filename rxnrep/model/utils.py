from typing import Callable, List, Optional, Union, Any

import numpy as np
import torch
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

    For hidden layers:
    Linear -> BN (default to False) -> Activation
    For output layer:
    Linear with the option to use bias of not



    Args:
        in_size: input feature size
        hidden_sizes: sizes for hidden layers
        batch_norm: whether to add 1D batch norm
        activation: activation function for hidden layers
        out_size: size of output layer
        out_bias: bias for output layer, this use set to False internally if
            out_batch_norm is used.
    """

    def __init__(
        self,
        in_size: int,
        hidden_sizes: List[int],
        *,
        batch_norm: bool = False,
        activation: Union[Callable, str] = "ReLU",
        out_size: Optional[int] = None,
        out_bias: bool = True,
    ):
        super().__init__()
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
        if out_size is not None:
            layers.append(nn.Linear(in_size, out_size, bias=out_bias))

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


def tensor_to_list(data: Any) -> Any:
    """
    Convert a tensor field in a data structure to list (list of list of ...).

    Args:
        data: data to convert, of type torch.Tensor, dict, list, tuple ...

    Returns:

        the same data structure, but with tensors converted.
    """
    if isinstance(data, torch.Tensor):
        return data.numpy().tolist()
    elif isinstance(data, tuple):
        return (tensor_to_list(v) for v in data)
    elif isinstance(data, list):
        return [tensor_to_list(v) for v in data]
    elif isinstance(data, dict):
        return {k: tensor_to_list(v) for k, v in data.items()}
    else:
        return data


def to_tensor(data: Any, dtype="float32") -> Any:
    """
    Convert floats, list of floats, or numpy array to tensors. The list and array can be
    placed in dictionaries.

    Args:
        data: data to convert
        dtype: data type of the tensor to convert

    Returns:
        The same data structure, but with list and array converted to tensor.
    """

    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)

    if isinstance(data, (float, list, np.ndarray)):
        return torch.as_tensor(data, dtype=dtype)
    elif isinstance(data, dict):
        return {k: to_tensor(v) for k, v in data.items()}
    else:
        return data