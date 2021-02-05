import logging
import os
import random
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import dgl
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

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
                print(
                    "EarlyStopping counter: {}/{}".format(self.counter, self.patience)
                )
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

    For example, the checkpoint saved by `save_checkpoints()`.

    Args:
        state_dict_objects: A dictionary of objects to load. The object should
            have load_state_dict() (e.g. model, optimizer, ...)
        map_location: device location to map the parameters
        filename: path to the saved checkpoint (e.g. by `save_checkpoint()`)
    """
    checkpoints = torch.load(str(filename), map_location)
    for k, obj in state_dict_objects.items():
        state_dict = checkpoints.pop(k)
        obj.load_state_dict(state_dict)
    misc_objects = checkpoints

    return misc_objects


def load_part_pretrained_model(model, map_location=None, filename="checkpoint.pkl"):
    """
    Load part of a pretrained model.

    Suppose pretrained model A has layers L1, L2, and L3, model B has layers L1, L2,
    L4, this function will load parameters of L1 and L2 from model A to model B.

    Take from: https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113


    Args:
        model: the model to load parameters in
        map_location: device location to map the parameters
        filename: path to the saved checkpoint (e.g. by `save_checkpoint()`)

    Returns:

    """
    checkpoints = torch.load(str(filename), map_location)
    pretrained_dict = checkpoints["model"]
    model_dict = model.state_dict()

    # filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)

    # load the new state dict
    model.load_state_dict(model_dict)


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

        # see also https://github.com/PyTorchLightning/pytorch-lightning/blob/0f64f15f525ae887bb1db7b643684d6f4e1b2d1e/pytorch_lightning/cluster_environments/slurm_environment.py#L21
        # use slurm job id for the port number
        # guarantees unique ports across jobs from same grid search
        default_port = os.environ["SLURM_JOB_ID"]
        default_port = default_port[-4:]
        default_port = int(default_port) + 15000  # all ports be in the 10k+ range

        root_node = os.environ["SLURM_NODELIST"].strip().split(",")[0]
        dist_url = f"tcp://{root_node}:{default_port}"

    # job started with spawn (should set args.rank and args.world_size)
    elif args.launch_mode == "spawn":
        try:
            assert args.rank is not None and args.rank >= 0
            assert args.world_size is not None and args.world_size >= 1
        except AttributeError as e:
            raise RuntimeError(
                "spawn launch mode needs `rank` and `world_size` be set in args"
            )

        dist_url = f"tcp://localhost:15678"
    else:
        raise ValueError("Not supported launch mode")

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
        backend=backend,
        init_method=dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )


class TimeMeter:
    """
    Measure running time.
    """

    def __init__(self, frequency=1):
        self.frequency = frequency
        self.t0 = time.time()
        self.t = self.t0

    def update(self):
        t = time.time()
        delta_t = t - self.t
        cumulative_t = t - self.t0
        self.t = t
        return delta_t, cumulative_t

    def display(self, counter, message=None, flush=False):
        t = time.time()
        delta_t = t - self.t
        cumulative_t = t - self.t0
        self.t = t

        if counter % self.frequency == 0:
            message = "\t\t" if message is None else f"\t\t{message} "
            message = message + " " * (45 - len(message))
            print(
                f"{message}delta time: {delta_t:.2f} cumulative time: {cumulative_t:.2f}",
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


def load_checkpoint_tensorboard(save_dir="./lightning_logs"):
    """
    Get the latest checkpoint path of tensorboard logger.
    """
    path = Path(save_dir).resolve()
    versions = os.listdir(path)
    v = sorted(versions)[-1]
    checkpoints = os.listdir(path.joinpath(f"{v}/checkpoints"))
    ckpt = sorted(checkpoints)[-1]

    ckpt_path = str(path.joinpath(f"{v}/checkpoints/{ckpt}"))

    return ckpt_path


def load_checkpoint_wandb(
    save_dir: Path, project: str, run_directory: str = "latest-run"
) -> Tuple[str, str]:
    """
    Get the latest checkpoint path and the identifier when using wandb logger.

    Args:
        save_dir: name of the directory to save wandb log, e.g. /some/path/wandb/
        project: project name of the wandb run
        run_directory: the directory for the run that stores files, logs, and run info,
            e.g. run-20210203_142512-6eooscnj
    Returns:
        ckpt_path: path to the latest run
        identifier: identifier of the wandb run
    """
    save_dir = Path(save_dir).expanduser().resolve()

    # get identifier of latest_run
    latest_run = save_dir.joinpath("wandb", run_directory).resolve()
    identifier = str(latest_run).split("-")[-1]

    # get checkpoint of latest run
    ckpt_path = save_dir.joinpath(
        project, identifier, "checkpoints", "last.ckpt"
    ).resolve()

    # the checkpoint does not exist
    if not latest_run.exists() or not ckpt_path.exists() or identifier == "run":
        ckpt_path = None
        identifier = None
    else:
        ckpt_path = str(ckpt_path)

    return ckpt_path, identifier


def get_repo_git_commit(repo_path: Path) -> str:
    """
    Get the latest git commit info of a github repository.

    Args:
        repo_path: path to the repo

    Returns:
        latest commit info
    """
    output = subprocess.check_output(["git", "log"], cwd=Path(repo_path))
    output = output.decode("utf-8").split("\n")[:6]
    latest_commit = "\n".join(output)
    return latest_commit


def save_files_to_wandb(wandb_logger, script: Path, other_files: List[str] = None):
    """
    Save files to wandb.

    Args:
        wandb_logger: lightning wandb logger
        script: the python script running the training to save
        other_files: name of other files in the same directory of the script. If the
            file does not exist, it is not saved to wandb and silently ignored.
    """
    wandb = wandb_logger.experiment

    wandb.save(script, policy="now")
    parent = Path(script).resolve().parent
    for f in other_files:
        if parent.joinpath(f).exists():
            wandb.save(str(parent.joinpath(f)), policy="now")
