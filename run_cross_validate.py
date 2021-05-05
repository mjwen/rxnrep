import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from rxnrep.train import train
from rxnrep.utils.cross_validate import compute_metric_statistics
from rxnrep.utils.hydra_config import (
    dump_hydra_config,
    get_datamodule_config,
    get_wandb_logger_config,
)
from rxnrep.utils.io import to_path

logger = logging.getLogger(__file__)

# HYDRA_FULL_ERROR=1 for complete stack trace
os.environ["HYDRA_FULL_ERROR"] = "1"


@hydra.main(config_path="configs", config_name="config_cross_validate.yaml")
def main(cfg: DictConfig):

    # Update cfg, new or modified ones by encoder and decoder
    # won't change the model behavior, only add some helper args
    cfg_update = hydra.utils.call(cfg.model.decoder.cfg_adjuster, cfg)

    # Reset CV filename to trainset filename of datamodule
    dm_cfg, _ = get_datamodule_config(cfg)
    cv_filename = OmegaConf.create(
        {"cross_validate": {"filename": dm_cfg.trainset_filename}}
    )
    cfg_update = OmegaConf.merge(cfg_update, cv_filename)

    # Merge cfg
    OmegaConf.set_struct(cfg, False)
    cfg_final = OmegaConf.merge(cfg, cfg_update)
    OmegaConf.set_struct(cfg_final, True)

    # Save configs to file
    dump_hydra_config(cfg, "hydra_cfg_original.yaml")
    dump_hydra_config(cfg_update, "hydra_cfg_update.yaml")
    dump_hydra_config(cfg_final, "hydra_cfg_final.yaml")

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
