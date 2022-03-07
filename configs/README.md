# RxnRep configuration

The directory contains the configuration files for the supervised training,
pretriaining, and finetuning tasks.
We use [hydra](https://github.com/facebookresearch/hydra) for all the
configurations.

It is already discussed in the main [README](../README.md) about how to run the
experiments for classification tasks and here are brief notes on how to run the
experiments for regression tasks.

## Train regression models

### Direct supervised training

To train for the `Heid` dataset:

```bash
python run.py --config-name config.yaml  datamodule=regression/heid.yaml \
       model/decoder=regressor.yaml
```

### Pretraining & finetuning

To pretrain the model using contrastive learning:

```bash
python run.py --config-name config_contrastive.yaml datamodule=contrastive/heid.yaml
```

Note, you may need to set `functional_group_smarts_filenames` in [subgraph.yaml](./transform1/subgraph.yaml) and [transform_or_identity.yaml](./transform1/transform_or_identity.yaml) to the path of [smarts_daylight.tsv](../assets/smarts_daylight.tsv) before running the above command, if you want to use the subgraph augmentation method.

To finetune the pretrained model:

```bash
python run.py --config-name config_finetune.yaml datamodule=regression/heid.yaml \
       datamodule.regression.allow_label_scaler_none=true \
       model/finetuner=regressor.yaml  pretrained_wandb_id=<wandb_id>
```

where `wandb_id` is the [W&B](https://wandb.ai) id for the pretraining run, an eight-character
alphanumeric
(e.g. `3oep187z`).
By providing the `wandb_id`, the finetuning script will automatically search for the pretrained model.

Alternatively, the pretrained model info can be passed in manually:

```bash
python run.py --config-name config_finetune.yaml datamodule=regression/heid.yaml \
       datamodule.regression.allow_label_scaler_none=true \
       model/finetuner=regressor.yaml \
       model.finetuner.model_class.pretrained_checkpoint_filename=<checkpoint> \
       model.finetuner.model_class.pretrained_dataset_state_dict_filename=<dataset_state_dict> \
       model.finetuner.model_class.pretrained_config_filename=<config>
```

where

- `checkpoint` is the path to the pretrained model checkpoint, e.g. `epoch=9-step=39.ckpt`
- `dataset_state_dict` is the path to the dataset state dict of the pretrained model, e.g. `dataset_state_dict.yaml`
- `config` is the path to the config file of the pretrained model, e.g. `hydra_cfg_final.yaml`
