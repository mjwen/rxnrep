import os

import hydra
from omegaconf import DictConfig, OmegaConf

# HYDRA_FULL_ERROR=1 for complete stack trace
os.environ["HYDRA_FULL_ERROR"] = "1"


@hydra.main(config_path="configs", config_name="config.yaml")
def main(cfg: DictConfig):

    # Put inside for autocompletion
    from rxnrep.train import train
    from rxnrep.utils.io import dump_hydra_config

    # Update cfg (new or modified ones)
    cfg_update = hydra.utils.call(cfg.model.decoder.cfg_adjuster, cfg)

    # Merge cfg
    OmegaConf.set_struct(cfg, False)
    cfg_merged = OmegaConf.merge(cfg, cfg_update)
    OmegaConf.set_struct(cfg, True)
    OmegaConf.set_struct(cfg_merged, True)

    # Save configs to file
    dump_hydra_config(cfg, "hydra_cfg_original.yaml")
    dump_hydra_config(cfg_update, "hydra_cfg_update.yaml")
    dump_hydra_config(cfg_merged, "hydra_cfg_final.yaml")

    # train the model
    train(cfg_merged)


if __name__ == "__main__":
    main()
