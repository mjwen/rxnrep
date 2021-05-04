import logging
from pathlib import Path
from typing import Union

from omegaconf import DictConfig, OmegaConf

from rxnrep.utils.io import to_path
from rxnrep.utils.wandb import (
    get_dataset_state_dict_latest_run,
    get_wandb_checkpoint_latest_run,
    get_wandb_identifier_latest_run,
)

logger = logging.getLogger(__file__)


def dump_hydra_config(cfg: DictConfig, filename: Union[str, Path]):
    """
    Save OmegaConfig to a yaml file.
    """
    with open(to_path(filename), "w") as f:
        OmegaConf.save(cfg, f, resolve=True)


def get_restore_config(config: DictConfig) -> DictConfig:
    """
    Get the config info used to restore the model from the latest run.

    This includes: dataset state dict path, checkpoint path, and wandb identifier.

    Args:
        config: hydra config

    Returns:
        DictConfig with info related to restoring the model.
    """

    # Get datamodule config
    # we do the for loop to get it because we group datamodule config into: predictive,
    # contrastive, finetune...
    for name in config.datamodule:
        dm_name = name
        dm_config = config.datamodule[name]
        break

    dataset_state_dict_filename = dm_config.get(
        "state_dict_filename", "dataset_state_dict.yaml"
    )
    project = config.logger.wandb.project
    path = to_path(config.original_working_dir).joinpath("outputs")

    dataset_state_dict = get_dataset_state_dict_latest_run(
        path, dataset_state_dict_filename
    )
    checkpoint = get_wandb_checkpoint_latest_run(path, project)
    identifier = get_wandb_identifier_latest_run(path)

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
