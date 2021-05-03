import logging
import os
import pickle
import random
from pathlib import Path
from typing import Union

import dgl
import numpy as np
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def to_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()


def create_directory(path: Union[str, Path], is_directory: bool = False):
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


def seed_all(seed=35, cudnn_benchmark=False, cudnn_deterministic=False):
    """
    Seed Python, numpy, torch, and dgl.

    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.deterministic = cudnn_deterministic

    dgl.random.seed(seed)


def dump_hydra_config(cfg: DictConfig, filename: Union[str, Path]):
    """
    Save OmegaConfig to a yaml file.
    """
    with open(to_path(filename), "w") as f:
        OmegaConf.save(cfg, f, resolve=True)
