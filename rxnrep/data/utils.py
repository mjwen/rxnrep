from typing import Any

import numpy as np
import torch


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