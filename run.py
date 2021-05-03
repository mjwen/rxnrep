import os

import hydra
from omegaconf import DictConfig, OmegaConf

# set HYDRA_FULL_ERROR for a complete stack trace
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["OC_CAUSE"] = "1"


@hydra.main(config_path="configs", config_name="config.yaml")
def main(cfg: DictConfig):

    from rxnrep.train import train
    from rxnrep.utils.adjust_config import print_config

    # Input config
    print_config(cfg, label="CONFIG (original)")

    # Update cfg (new or modified ones)
    cfg_updated = hydra.utils.call(cfg.model.decoder.cfg_adjuster, cfg)
    print("\n\n\n")
    print_config(cfg_updated, label="CONFIG (internally modified or added)")

    # Merge cfg
    OmegaConf.set_struct(cfg, False)
    cfg_merged = OmegaConf.merge(cfg, cfg_updated)
    OmegaConf.set_struct(cfg, True)
    OmegaConf.set_struct(cfg_merged, True)
    print("\n\n\n")
    print_config(cfg_merged, label="CONFIG (final)")

    # train the model
    train(cfg_merged)


if __name__ == "__main__":
    main()
