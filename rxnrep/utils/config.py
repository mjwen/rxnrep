import logging
from pathlib import Path
from typing import Tuple, Union

import rich
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from rich.syntax import Syntax
from rich.tree import Tree

from rxnrep.utils.io import to_path
from rxnrep.utils.wandb import (
    get_dataset_state_dict_latest_run,
    get_wandb_checkpoint_latest_run,
    get_wandb_identifier_latest_run,
)

logger = logging.getLogger(__file__)


def get_datamodule_config(config: DictConfig) -> Tuple[DictConfig, str]:
    """
    Get datamodule config.

    Args:
        config: main config

    Returns:
        dm_config: datamodule config
        dm_name: name of the datamodule (directory name in configs/datamodule). For
            example, `predictive`, `contrastive`.
    """

    # There is essentially one datamodule config; but since we group datamodule config
    # into predictive, contrastive, finetune... we do not know the datamodule name.
    # So, we first get the name and then get the module.

    dm_name = list(config.datamodule.keys())[0]
    dm_config = config.datamodule[dm_name]

    return dm_config, dm_name


def get_wandb_logger_config(config: DictConfig) -> Union[DictConfig, None]:
    """
    Get wandb logger config.

    Args:
        config: main config.

    Returns:
        wandb logger config. None if no wandb logger is used.
    """
    logger_config = None
    for name in config.logger:
        if name == "wandb":
            logger_config = config.logger[name]

    return logger_config


def get_restore_config(config: DictConfig) -> DictConfig:
    """
    Get the config info used to restore the model from the latest run.

    This includes: dataset state dict path, checkpoint path, and wandb identifier.

    Args:
        config: hydra config

    Returns:
        DictConfig with info related to restoring the model.
    """

    dm_config, dm_name = get_datamodule_config(config)

    dataset_state_dict_filename = dm_config.get(
        "state_dict_filename", "dataset_state_dict.yaml"
    )
    project = config.logger.wandb.project
    path = to_path(config.original_working_dir).joinpath("outputs")

    dataset_state_dict = get_dataset_state_dict_latest_run(
        path, dataset_state_dict_filename, index=-2
    )
    checkpoint = get_wandb_checkpoint_latest_run(path, project, index=-2)
    identifier = get_wandb_identifier_latest_run(path, index=-2)

    d = {
        "datamodule": {dm_name: {"restore_state_dict_filename": dataset_state_dict}},
        "callbacks": {"wandb": {"id": identifier}},
        "trainer": {"resume_from_checkpoint": checkpoint},
    }

    logger.info(f"Restoring training with automatically determined info: {d}")

    if dataset_state_dict is None:
        logger.warning(
            f"Trying to automatically restore dataset state dict, but cannot find latest "
            f"dataset state dict file. Now, we set it to `None` to compute dataset "
            f"statistics (e.g. feature mean and standard deviation) from the trainset."
        )
    if checkpoint is None:
        logger.warning(
            f"Trying to automatically restore model from checkpoint, but cannot find "
            f"latest checkpoint file. Proceed without restoring."
        )
    if identifier is None:
        logger.warning(
            f"Trying to automatically restore training with the same wandb identifier, "
            f"but cannot find the identifier of latest run. A new wandb identifier will "
            f"be assigned."
        )

    restore_config = OmegaConf.create(d)

    return restore_config


def merge_configs(*configs: DictConfig) -> DictConfig:
    """
    Merge multi configs into one.

    The `struct` flag of input configs will not be altered.

    Return:
        merged config
    """
    assert len(configs) > 1, f"Expect more than 1 config to merge, got {len(configs)}"

    # set all struct info to False
    is_struct = []
    for c in configs:
        s = OmegaConf.is_struct(c)
        is_struct.append(s)
        if s:
            OmegaConf.set_struct(c, False)

    merged = OmegaConf.merge(*configs)

    # restore struct info
    for c, s in zip(configs, is_struct):
        if s:
            OmegaConf.set_struct(c, True)

    return merged


def dump_config(config: DictConfig, filename: Union[str, Path]):
    """
    Save OmegaConfig to a yaml file.
    """
    with open(to_path(filename), "w") as f:
        OmegaConf.save(config, f, resolve=True)


@rank_zero_only
def print_config(
    config: DictConfig,
    label: str = "CONFIG",
    resolve: bool = True,
    sort_keys: bool = True,
):
    """
    Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config: Config.
        label: label of the config.
        resolve: whether to resolve reference fields of DictConfig.
        sort_keys: whether to sort config keys.
    """

    tree = Tree(f":gear: {label}")

    for field, config_section in config.items():
        branch = tree.add(field)

        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(
                config_section, resolve=resolve, sort_keys=sort_keys
            )
        else:
            branch_content = str(config_section)

        branch.add(Syntax(branch_content, "yaml"))

    rich.print(tree)
