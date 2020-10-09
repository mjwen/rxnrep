import random
import os
import shutil
import logging
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import dgl
from typing import Any, Dict


logger = logging.getLogger(__name__)


class EarlyStopping:
    def __init__(self, patience=200, silent=True):
        self.patience = patience
        self.silent = silent
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if not self.silent:
                print("EarlyStopping counter: {}/{}".format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def seed_all(seed=35, cudnn_benchmark=False, cudnn_deterministic=False):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.deterministic = cudnn_deterministic
    dgl.random.seed(seed)


def save_checkpoints(
    state_dict_objects, misc_objects, is_best, msg=None, filename="checkpoint.pkl"
):
    """
    Save checkpoints for all objects for later recovery.

    Args:
        state_dict_objects (dict): A dictionary of objects to save. The object should
            have state_dict() (e.g. model, optimizer, ...)
        misc_objects (dict): plain python object to save
        filename (str): filename for the checkpoint

    """
    objects = misc_objects.copy()
    for k, obj in state_dict_objects.items():
        objects[k] = obj.state_dict()
    torch.save(objects, filename)
    if is_best:
        shutil.copyfile(filename, "best_checkpoint.pkl")
        if msg is not None:
            logger.info(msg)


def load_checkpoints(state_dict_objects, map_location=None, filename="checkpoint.pkl"):
    """
    Load checkpoints for all objects for later recovery.

    Args:
        state_dict_objects (dict): A dictionary of objects to save. The object should
            have load_state_dict() (e.g. model, optimizer, ...)
    """
    checkpoints = torch.load(str(filename), map_location)
    for k, obj in state_dict_objects.items():
        state_dict = checkpoints.pop(k)
        obj.load_state_dict(state_dict)
    misc_objects = checkpoints

    return misc_objects


def init_distributed_mode(args):
    """
    Initialize the following variables:
        - world_size
        - rank
        - gpu_to_work_on
    """

    args.is_slurm_job = "SLURM_JOB_ID" in os.environ

    if args.is_slurm_job:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.world_size = int(os.environ["SLURM_NNODES"]) * int(
            os.environ["SLURM_TASKS_PER_NODE"][0]
        )
    else:
        # multi-GPU job (local or multi-node) - jobs started with torch.distributed.launch
        # read environment variables
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])

    # prepare distributed
    if args.gpu:
        backend = "nccl"
        gpu_id = args.rank % torch.cuda.device_count()
        args.device = torch.device("cuda", gpu_id)
    else:
        backend = "gloo"
        args.device = torch.device("cpu")

    dist.init_process_group(
        backend=backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )


class ProgressMeter:
    """
    Log stuff with pandas
    """

    def __init__(self, path, restore=False):
        self.path = path

        # reload path stats
        if os.path.isfile(self.path) and restore:
            self.stats = pd.read_csv(self.path, index_col=None)
        else:
            self.stats = None

    def update(self, row: Dict[str, Any], save=True):
        if self.stats is None:
            self.stats = pd.DataFrame([row])
        else:
            self.stats = self.stats.append(row, ignore_index=True)

        # save the statistics
        if save:
            self.stats.to_csv(self.path, index=False)

    def display(self):
        # get the last row as a dict
        latest = self.stats.tail(n=1).to_dict("records")[0]

        fmt_str = ["{}: {:.2f}".format(k, v) for k, v in latest.items()]
        fmt_str = ", ".join(fmt_str)
        print(fmt_str)
