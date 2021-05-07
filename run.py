import os

import hydra
from omegaconf import DictConfig, OmegaConf

from rxnrep.train import train
from rxnrep.utils.config import (
    dump_config,
    get_restore_config,
    merge_configs,
    print_config,
)
from rxnrep.utils.io import to_path
from rxnrep.utils.wandb import copy_pretrained_model

# HYDRA_FULL_ERROR=1 for complete stack trace
os.environ["HYDRA_FULL_ERROR"] = "1"


@hydra.main(config_path="configs", config_name="config.yaml")
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

    # Restore cfg from latest run (restore_dataset_state_dict, checkpoint, and wandb_id)
    if cfg.restore:
        cfg_restore = get_restore_config(cfg)
        # WARNING, order matters in the below line
        # cfg_update and cfg_restore can both update `restore_dataset_state_dict`;
        # cfg_restore comes after since restored value has higher precedence.
        # Don't worry, cfg_restore is empty if no model is found to restore,
        # so it won't ruin the value in cfg_update.
        cfg_update = merge_configs(cfg_update, cfg_restore)

    # Merge cfg
    cfg_final = merge_configs(cfg, cfg_update)
    OmegaConf.set_struct(cfg_final, True)

    # Save configs to file
    dump_config(cfg, "hydra_cfg_original.yaml")
    dump_config(cfg_update, "hydra_cfg_update.yaml")
    dump_config(cfg_final, "hydra_cfg_final.yaml")

    # It does not bother to print it again, useful for debug
    print_config(cfg_final, label="CONFIG", resolve=True, sort_keys=True)

    # train the model
    train(cfg_final)


if __name__ == "__main__":
    main()
