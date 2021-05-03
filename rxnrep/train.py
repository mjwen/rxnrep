import logging
from typing import List, Optional

import hydra
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase, WandbLogger

from rxnrep.utils.wandb import save_files_to_wandb, write_running_metadata

logger = logging.getLogger(__file__)


def train(config: DictConfig) -> Optional[float]:
    """
    Contains training pipeline.

    Instantiates all PyTorch Lightning objects from config.

    Args:
        config: Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed)

    # Init Lightning datamodule
    if "contrastive" in config.datamodule:
        dm = config.datamodule.contrastive
    elif "predictive" in config.datamodule:
        dm = config.datamodule.predictive
    else:
        raise RuntimeError(f"Unsupported datamodule {config.datamodule}")

    logger.info(f"Instantiating datamodule: {dm._target_}")
    datamodule: LightningDataModule = hydra.utils.instantiate(
        dm, restore=config.restore
    )

    # manually call them to get data to set up the model
    # (Lightning still ensures the method runs on the correct devices)
    datamodule.prepare_data()
    datamodule.setup()
    logger.info(f"Finished instantiating datamodule: {dm._target_}")

    # datamodule info passed to model
    dataset_info = datamodule.get_to_model_info()

    # Init Lightning model
    logger.info(f"Instantiating model: {config.model.decoder.model_class._target_}")

    # note encoder only provides args, decoder has the actual _target_
    encoder = config.model.encoder
    model: LightningModule = hydra.utils.instantiate(
        config.model.decoder.model_class,
        dataset_info=dataset_info,
        **encoder,
        **config.optimizer,
    )

    # Init Lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                logger.info(f"Instantiating callback: {cb_conf._target_}")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init Lightning loggers
    lit_logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                logger.info(f"Instantiating logger: {lg_conf._target_}")
                lit_logger.append(hydra.utils.instantiate(lg_conf))

    # Init Lightning trainer
    logger.info(f"Instantiating trainer: {config.trainer._target_}")
    trainer: Trainer = hydra.utils.instantiate(
        #     config.trainer, callbacks=callbacks, logger=lit_logger, _convert_="partial"
        config.trainer,
        callbacks=callbacks,
        logger=lit_logger,
    )

    # Train the model
    logger.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    # Evaluate model on test set after training
    if not config.get("skip_test", None):
        logger.info("Starting testing!")
        trainer.test()

    # Save additional files to wandb
    wandb_logger = None
    for ll in lit_logger:
        if isinstance(ll, WandbLogger):
            wandb_logger = ll
            break

    if wandb_logger:
        running_meta = "running_metadata.yaml"
        write_running_metadata(config.get("git_repo_path", None), filename=running_meta)

        files_to_save = [
            running_meta,
            "dataset_state_dict.yaml",
            "hydra_cfg_original.yaml",
            "hydra_cfg_update.yaml",
            "hydra_cfg_final.yaml",
            "run.log",
            # might exist
            "submit.sh",
            "sweep.py",
        ]

        logger.info(f"Saving extra files to wandb: {', '.join(files_to_save)}")
        save_files_to_wandb(wandb_logger, files_to_save)

    # Print path to best checkpoint
    logger.info(f"Best checkpoint path: {trainer.checkpoint_callback.best_model_path}")

    logger.info("Finalizing!")

    # Return metric score
    optimized_metric = config.get("optimized_metric", None)
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric]
