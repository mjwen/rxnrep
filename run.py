import os

import hydra
from omegaconf import DictConfig, OmegaConf

# set HYDRA_FULL_ERROR for a complete stack trace
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["OC_CAUSE"] = "1"


@hydra.main(config_path="configs", config_name="config.yaml")
def main(cfg: DictConfig):

    from rxnrep.train import train

    print(OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True))

    # enable adding new keys to config
    OmegaConf.set_struct(cfg, False)
    # adjust configs
    hydra.utils.call(cfg.model.decoder.cfg_adjuster, cfg)
    # disable adding new keys to config
    OmegaConf.set_struct(cfg, True)

    # TODO add config adjuster modifier here

    print(f"Current working directory : {os.getcwd()}")
    print(f"Orig working directory    : {hydra.utils.get_original_cwd()}")

    train(cfg)


if __name__ == "__main__":
    main()
