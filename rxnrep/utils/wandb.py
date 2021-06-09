import glob
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Union

from pytorch_lightning.utilities import rank_zero_only

from rxnrep.utils.io import create_directory, to_path, yaml_dump

logger = logging.getLogger(__name__)


def get_git_repo_commit(repo_path: Path) -> str:
    """
    Get the latest git commit info of a github repository.

    Args:
        repo_path: path to the repo

    Returns:
        latest commit info
    """
    output = subprocess.check_output(["git", "log"], cwd=to_path(repo_path))
    output = output.decode("utf-8").split("\n")[:6]
    latest_commit = "\n".join(output)

    return latest_commit


def get_hostname() -> str:
    """
    Get the hostname of the machine.

    Returns:
        hostname
    """
    output = subprocess.check_output("hostname")
    hostname = output.decode("utf-8").strip()

    return hostname


def write_running_metadata(
    git_repo: Optional[Path] = None, filename: str = "running_metadata.yaml"
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

    d = {"running_dir": Path.cwd().as_posix(), "hostname": get_hostname()}
    if git_repo is not None:
        d["git_commit"] = get_git_repo_commit(git_repo)

    yaml_dump(d, filename)


@rank_zero_only
def save_files_to_wandb(wandb_logger, files: List[str] = None):
    """
    Save files to wandb. The files should be given relative to cwd.

    Args:
        wandb_logger: lightning wandb logger
        files: name of the files in the running directory to save. If a file does not
            exist, it is silently ignored.
    """
    wandb = wandb_logger.experiment

    for f in files:
        p = Path.cwd().joinpath(f)
        if p.exists():
            wandb.save(p.as_posix(), policy="now", base_path=".")


def get_hydra_latest_run(path: Union[str, Path], index: int = -1) -> Union[Path, None]:
    """
    Find the latest hydra running directory in the hydra `outputs`.

    This assumes the hydra outputs look like:

    - outputs
      - 2021-05-02
        - 11-26-08
        - 12-01-19
      - 2021-05-03
        - 09-08-01
        - 10-10-01

    Args:
        path: path to the `outputs` directory. For example, this should be `../../`
        relative to the hydra current working directory cwd.
        index: index to the hydra runs to return. By default, -1 returns the last one.
            But this may not be the one we want when we are calling this from a hydra
            run, since -1 is the index to itself. In this case, set index to -2 to
            get the latest one before the current one.

    Returns:
        Path to the latest hydra run. `None` if not such path can be found.
    """

    path = to_path(path)
    all_paths = []
    for date in os.listdir(path):
        for time in os.listdir(path.joinpath(date)):
            all_paths.append(path.joinpath(date, time))
    all_paths = sorted(all_paths, key=lambda p: p.as_posix())

    if len(all_paths) < abs(index):
        return None
    else:
        return all_paths[index]


def get_dataset_state_dict_latest_run(
    path: Union[str, Path], name: str, index: int = -1
) -> Union[str, None]:
    """
    Get path to the dataset state dict of the latest run.

    Args:
        path: path to hydra `outputs` directory. For example, this should be `../../`
            relative to the hydra current working directory cwd.
        name: name of the dataset state dict file, e.g. dataset_state_dict.yaml
        index: index to the hydra runs to return. By default, -1 returns the last one.
            But this may not be the one we want when we are calling this from a hydra
            run, since -1 is the index to itself. In this case, set index to -2 to
            get the latest one before the current one.

    Returns:
        path to the dataset state dict yaml file. None if cannot find the file
    """
    latest = get_hydra_latest_run(path, index=index)

    if latest is not None:
        dataset_state_dict = latest.joinpath(name)
        if dataset_state_dict.exists():
            return dataset_state_dict.as_posix()

    return None


def get_wandb_identifier_latest_run(
    path: Union[str, Path], index: int = -1
) -> Union[str, None]:
    """
    Get the wandb unique identifier of the latest run.

    This assumes wandb logger `save_dir` is set to `.`, i.e. relative to hydra working
    directory.

    Args:
        path: path to hydra `outputs` directory. For example, this should be `../../`
            relative to the hydra current working directory cwd.
        index: index to the hydra runs to return. By default, -1 returns the last one.
            But this may not be the one we want when we are calling this from a hydra
            run, since -1 is the index to itself. In this case, set index to -2 to
            get the latest one before the current one.

    Returns:
        identifier str, None if cannot find the run
    """
    latest = get_hydra_latest_run(path, index=index)

    if latest is not None:
        latest_run = latest.joinpath("wandb", "latest-run").resolve()
        if latest_run.exists():
            identifier = str(latest_run).split("-")[-1]
            if identifier != "run":
                return identifier

    return None


# TODO this can be written such that project is not needed
def get_wandb_checkpoint_latest_run(
    path: Union[str, Path], project: str, index: int = -1
) -> Union[str, None]:
    """
    Get the wandb checkpoint of the latest run.

    Args:
        path: path to hydra `outputs` directory. For example, this should be `../../`
            relative to the hydra current working directory cwd.
        project: project name of the wandb run
        index: index to the hydra runs to return. By default, -1 returns the last one.
            But this may not be the one we want when we are calling this from a hydra
            run, since -1 is the index to itself. In this case, set index to -2 to
            get the latest one before the current one.

    Returns:
        path to the latest checkpoint, None if it does not exist
    """
    latest = get_hydra_latest_run(path, index=index)

    if latest is not None:
        identifier = get_wandb_identifier_latest_run(path, index=index)
        if identifier is not None:
            checkpoint = latest.joinpath(
                project, identifier, "checkpoints", "last.ckpt"
            ).resolve()
            if checkpoint.exists():
                return checkpoint.as_posix()

    return None


def get_wandb_run_path(identifier: str, path: Union[str, Path] = "."):
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


def get_wandb_checkpoint_path(identifier: str, path: Union[str, Path] = "."):
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


def copy_pretrained_model(
    identifier: str,
    source_dir: Union[str, Path] = ".",
    target_dir: Path = "pretrained_model",
    last_checkpoint: bool = False,
):
    """
    Copy the checkpoint and dataset_state_dict.yaml to a directory.

    Args:
        identifier: wandb unique identifier of experiment, e.g. 2i3rocdl
        source_dir:
        target_dir:
        last_checkpoint: whether to copy the last checkpoint name `last.ckpt` or the
            best validation checkpoints. If `False`, will find the checkpoint with the
            largest epoch number in something like: `epoch=92-step=200.ckpt`,
            `epoch=121-step=300.ckpt`... and copy it.
    """
    # create target dir
    target_dir = to_path(target_dir)
    create_directory(target_dir, is_directory=True)

    # copy checkpoint file
    ckpt_dir = get_wandb_checkpoint_path(identifier, source_dir)
    print("Checkpoint path:", ckpt_dir)

    if last_checkpoint:
        ckpt = os.path.join(ckpt_dir, "last.ckpt")
    else:
        checkpoints = glob.glob(os.path.join(ckpt_dir, "epoch=*.ckpt"))

        # find largest epoch number (Note, cannot sort the string directly, which may give
        # wrong results: e.g. `epoch=92-step=200.ckpt` will come after
        # `epoch=121-step=300.ckpt` if simply sort by string.
        largest_epoch = 0
        for p in checkpoints:
            name = Path(p).name
            epoch = int(name.split("-")[0].split("=")[1])
            if epoch > largest_epoch:
                largest_epoch = epoch
        ckpt = glob.glob(os.path.join(ckpt_dir, f"epoch={largest_epoch}*.ckpt"))[0]

    shutil.copy(ckpt, target_dir.joinpath("checkpoint.ckpt"))

    run_path = get_wandb_run_path(identifier, source_dir)
    print("wandb run path:", run_path)

    # copy hydra_cfg_final.yaml file
    f = to_path(run_path).joinpath("files", "hydra_cfg_final.yaml").resolve()
    shutil.copy(f, target_dir.joinpath("hydra_cfg_final.yaml"))

    # copy dataset state dict
    f = to_path(run_path).joinpath("files", "dataset_state_dict.yaml").resolve()
    shutil.copy(f, target_dir.joinpath("dataset_state_dict.yaml"))


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
