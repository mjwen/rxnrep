import logging
import os
import pickle
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


def to_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()


def create_directory(path: Union[str, Path], is_directory=False):
    p = to_path(path)
    if is_directory:
        dirname = p
    else:
        dirname = p.parent
    if not dirname.exists():
        os.makedirs(dirname)


def yaml_dump(obj, filename):
    create_directory(filename)
    with open(to_path(filename), "w") as f:
        yaml.dump(obj, f, default_flow_style=False)


def yaml_load(filename):
    with open(to_path(filename), "r") as f:
        obj = yaml.safe_load(f)
    return obj


def pickle_dump(obj, filename):
    create_directory(filename)
    with open(to_path(filename), "wb") as f:
        pickle.dump(obj, f)


def pickle_load(filename):
    with open(to_path(filename), "rb") as f:
        obj = pickle.load(f)
    return obj


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
    Convert list of floats or numpy array to tensors. The list and array can be placed
    in dictionaries.

    Args:
        data: data to convert
        dtype: data type of the tensor to convert

    Returns:
        The same data structure, but with list and array converted to tensor.
    """

    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)

    if isinstance(data, list):
        return torch.as_tensor(data, dtype=dtype)
    elif isinstance(data, np.ndarray):
        return torch.as_tensor(data, dtype=dtype)
    elif isinstance(data, dict):
        return {k: to_tensor(v) for k, v in data.items()}
    else:
        return data
