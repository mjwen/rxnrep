import random
import os
import shutil
import logging
import numpy as np
import pandas as pd
import time
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

    # jobs started with torch.distributed.launch
    if args.launch_mode == "torch_launch":
        # read environment variables
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        dist_url = f"tcp://localhost:15678"

    # job started with srun
    elif args.launch_mode == "srun":
        args.rank = int(os.environ["SLURM_PROCID"])
        args.world_size = int(os.environ["SLURM_NNODES"]) * int(
            os.environ["SLURM_TASKS_PER_NODE"][0]
        )
        # Based on pytorch-lightning setup at:
        # https://github.com/PyTorchLightning/pytorch-lightning/blob/2b255a3df4e20911c5a3704e8004a6c6958f8dfc/pytorch_lightning/trainer/connectors/slurm_connector.py

        # use slurm job id for the port number
        # guarantees unique ports across jobs from same grid search
        default_port = os.environ["SLURM_JOB_ID"]
        default_port = default_port[-4:]
        default_port = int(default_port) + 15000  # all ports should be in the 10k+ range
        root_node = os.environ["SLURM_NODELIST"].strip().split(",")[0]
        dist_url = f"tcp://{root_node}:{default_port}"

    # job started with spawn (should set args.rank and args.world_size)
    else:
        dist_url = f"tcp://localhost:15678"

    # override dist_url is provided in args
    if args.dist_url is not None:
        dist_url = args.dist_url

    # prepare distributed
    if args.gpu:
        backend = "nccl"
        gpu_id = args.rank % torch.cuda.device_count()
        args.device = torch.device("cuda", gpu_id)
    else:
        backend = "gloo"
        args.device = torch.device("cpu")

    logger.info(
        f"dist.init_process_group parameters: backend({backend}), "
        f"init_method({dist_url}), world_size({args.world_size})."
    )
    dist.init_process_group(
        backend=backend, init_method=dist_url, world_size=args.world_size, rank=args.rank
    )


class TimeMeter:
    """
    Measure running time.
    """

    def __init__(self, frequency=1):
        self.frequency = frequency
        self.t0 = time.time()
        self.t = self.t0

    def display(self, counter, message=None, flush=False):
        t = time.time()
        delta_t = t - self.t
        cumulative_t = t - self.t0
        self.t = t

        if counter % self.frequency == 0:
            message = "\t\t" if message is None else f"\t\t{message}; "
            message = message + " " * (45 - len(message))
            print(
                "{}delta time: {:.2f} cumulative time: {:.2f}".format(
                    message, delta_t, cumulative_t
                ),
                flush=flush,
            )

        return delta_t, cumulative_t


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
