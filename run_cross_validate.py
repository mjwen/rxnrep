"""
Cross validation.

Data split:
- the dataset for cv should be provided as dm.trainset_filename; dm.valset_fileanme is
  always ignored; and dm.testset_filename is used if provided
- the dataset provided via dm.trainset_filename is split into kfolds of train set and
  test set
- for each fold, the train set is reassigned to dm.trainset_filename and the test set is
  assigned to dm.valset_filename
- if dm.testset_filename is provided, it is used for test; if not, the test set in each
  fold is assigned to dm.testset_filename (then in each fold the val set and the test set
  are the same)
"""
import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from rxnrep.train import train
from rxnrep.utils.config import (
    dump_config,
    get_datamodule_config,
    get_wandb_logger_config,
    merge_configs,
    print_config,
)
from rxnrep.utils.cross_validate import compute_metric_statistics
from rxnrep.utils.io import to_path
from rxnrep.utils.wandb import copy_pretrained_model

logger = logging.getLogger(__file__)

# HYDRA_FULL_ERROR=1 for complete stack trace
os.environ["HYDRA_FULL_ERROR"] = "1"


@hydra.main(config_path="configs", config_name="config_cross_validate.yaml")
def main(cfg: DictConfig):

    # The copy_trained_model fn here is only for test purpose, should remove
    if "finetuner" in cfg.model:
        wandb_id = cfg.get("pretrained_wandb_id", None)
        if wandb_id:
            path = to_path(cfg.original_working_dir).joinpath("outputs")
            copy_pretrained_model(wandb_id, source_dir=path)

    # Update cfg, new or modified ones by model encoder and decoder
    # won't change the model behavior, only add some helper args
    if "finetuner" in cfg.model:
        cfg_update = hydra.utils.call(cfg.model.finetuner.cfg_adjuster, cfg)
    else:
        cfg_update = hydra.utils.call(cfg.model.decoder.cfg_adjuster, cfg)

    # Reset CV filename to trainset filename of datamodule
    dm_cfg, _ = get_datamodule_config(cfg)
    cv_filename = OmegaConf.create(
        {"cross_validate": {"filename": dm_cfg.trainset_filename}}
    )
    cfg_update = merge_configs(cfg_update, cv_filename)

    # Merge cfg
    cfg_final = merge_configs(cfg, cfg_update)
    OmegaConf.set_struct(cfg_final, True)

    # Save configs to file
    dump_config(cfg, "hydra_cfg_original.yaml")
    dump_config(cfg_update, "hydra_cfg_update.yaml")
    dump_config(cfg_final, "hydra_cfg_final.yaml")

    # It does not bother to print it again, useful for debug
    print_config(cfg_final, label="CONFIG", resolve=True, sort_keys=True)

    # Get CV data split

    data_splits = hydra.utils.call(cfg_final.cross_validate)

    # Determine whether testset_filename is provided in datamodule
    # (should not do this in the loop since dm_cfg.testset_filename is reset)
    dm_cfg, _ = get_datamodule_config(cfg_final)
    if dm_cfg.testset_filename:
        dm_has_testset = True
    else:
        dm_has_testset = False

    metric_scores = []
    for i, (trainset, testset) in enumerate(data_splits):

        OmegaConf.set_struct(cfg_final, False)

        # Update datamodule (trainset_filename, valset_filename, and testset_filename)
        dm_cfg.trainset_filename = str(trainset)
        dm_cfg.valset_filename = str(testset)
        if not dm_has_testset:
            dm_cfg.testset_filename = str(testset)

        # Update wandb logger info (save_dir)
        wandb_logger_cfg = get_wandb_logger_config(cfg_final)
        wandb_save_dir = str(to_path(trainset).parent)
        wandb_logger_cfg.save_dir = wandb_save_dir

        OmegaConf.set_struct(cfg_final, True)

        logger.info(
            f"Cross validation fold {i}. Set wandb save_dir to: {wandb_save_dir}"
        )
        logger.info(
            f"Cross validation fold {i}. With trainset: {dm_cfg.trainset_filename}, "
            f"valset: {dm_cfg.valset_filename}, and testset: {dm_cfg.testset_filename}"
        )

        # train the model
        score = train(cfg_final)
        metric_scores.append(score)

    metrics, mean, std = compute_metric_statistics(metric_scores)
    logger.info(f"Cross validation metrics (all): {metrics}")
    logger.info(f"Cross validation metrics (mean): {mean}")
    logger.info(f"Cross validation metrics (std): {std}")


if __name__ == "__main__":
    main()
