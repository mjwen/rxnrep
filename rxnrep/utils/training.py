import glob
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from rxnrep.utils.io import create_directory, to_path, yaml_dump

logger = logging.getLogger(__name__)


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
        save_dir: name of the directory to save wandb logger, e.g. /some/path/wandb/
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
    output = subprocess.check_output(["git", "logger"], cwd=Path(repo_path))
    output = output.decode("utf-8").split("\n")[:6]
    latest_commit = "\n".join(output)
    return latest_commit


def write_running_metadata(
    filename: str = "running_metadata.yaml", git_repo: Optional[Path] = None
):
    """
    Write additional running metadata to a file and then copy it to wandb.

    Currently, we write:
    - the running dir, i.e. cwd
    - git repo commit, optional

    Args:
        filename: name of the file to write
        git_repo: path to the git repo, if None, do not use this info.
    """

    d = {"running_dir": Path.cwd().as_posix()}
    if git_repo is not None:
        d["git_commit"] = get_repo_git_commit(git_repo)

    yaml_dump(d, filename)


def save_files_to_wandb(wandb_logger, files: List[str] = None):
    """
    Save files to wandb.

    Args:
        wandb_logger: lightning wandb logger
        files: name of the files in the running directory to save. If a file does not
            exist, it is silently ignored.
    """
    wandb = wandb_logger.experiment

    for f in files:
        fname = Path.cwd().joinpath(f)
        if fname.exists():
            wandb.save(str(fname), policy="now", base_path="../scripts")


def get_wandb_run_path(identifier: str, path="."):
    """
    Args:
        identifier: wandb unique identifier of experiment, e.g. 2i3rocdl
        path: root path to search
    Returns:
        path to the wandb run directory:
        e.g. running_dir/job_0/wandb/wandb/run-20201210_160100-3kypdqsw
    """
    for root, dirs, files in os.walk(path):
        if "wandb" not in root:
            continue
        for d in dirs:
            if d.startswith("run-") or d.startswith("offline-run-"):
                if d.split("-")[-1] == identifier:
                    return os.path.abspath(os.path.join(root, d))

    raise RuntimeError(f"Cannot found job {identifier} in {path}")


def get_wandb_checkpoint_path(identifier: str, path="."):
    """
    Args:
        identifier: wandb unique identifier of experiment, e.g. 2i3rocdl
        path: root path to search
    Returns:
        path to the wandb checkpoint directory:
        e.g. running_dir/job_0/wandb/<project_name>/<identifier>/checkpoints
    """
    for root, dirs, files in os.walk(path):
        if root.endswith(f"{identifier}/checkpoints"):
            return os.path.abspath(root)

    raise RuntimeError(f"Cannot found job {identifier} in {path}")


def copy_trained_model(
    identifier: str, source_dir: Path = ".", target_dir: Path = "trained_model"
):
    """
    Copy the last checkpoint and dataset_state_dict.yaml to a directory.

    Args:
        identifier: wandb unique identifier of experiment, e.g. 2i3rocdl
        source_dir:
        target_dir:
    """
    # create target dir
    target_dir = to_path(target_dir)
    create_directory(target_dir, is_directory=True)

    # copy checkpoint file
    ckpt_dir = get_wandb_checkpoint_path(identifier, source_dir)
    print("Checkpoint path:", ckpt_dir)

    checkpoints = glob.glob(os.path.join(ckpt_dir, "epoch=*.ckpt"))
    checkpoints = sorted(checkpoints)
    shutil.copy(checkpoints[-1], target_dir.joinpath("checkpoint.ckpt"))

    # copy config.yaml file
    run_path = get_wandb_run_path(identifier, source_dir)
    print("wandb run path:", run_path)

    f = to_path(run_path).joinpath("files", "config.yaml")
    shutil.copy(f, target_dir.joinpath("config.yaml"))

    # copy dataset state dict
    f = to_path(run_path).joinpath("files", "../scripts/dataset_state_dict.yaml")
    shutil.copy(f, target_dir.joinpath("dataset_state_dict.yaml"))
