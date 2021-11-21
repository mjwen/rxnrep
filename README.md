# RxnRep

Self-supervised contrastive pretraining for chemical reaction representation (RxnRep).

<p align="center">
<img src="rxnrep.png" alt="rxnrep" width="600">
</p>

## Installation

```bash
git clone https://github.com/mjwen/rxnrep.git
cd rxnrep
conda env create -f environment.yml
conda activate rxnrep
pip install -e .
```

## Train the model

### Pretraining & finetuning

To pretrain the model using contrastive learning:

```bash
python run.py --config-name config_contrastive.yaml datamodule=contrastive/schneider.yaml
```

To finetune the pretrained model:

```bash
python run.py --config-name config_finetune.yaml datamodule=classification/schneider.yaml \
    pretrained_wandb_id=<wandb_id>
```

where `wandb_id` is the W&B id for the pretraining run, an eight-character alphanumeric
(e.g. `3oep187z`).
By providing the `wandb_id`, the finetuning script will automatically search for the pretrained model.

Alternatively, the pretrained model info can be passed in manually:

```bash
python run.py --config-name config_finetune.yaml datamodule=classification/schneider.yaml \
    model.finetuner.model_class.pretrained_checkpoint_filename=<checkpoint> \
    model.finetuner.model_class.pretrained_dataset_state_dict_filename=<dataset_state_dict> \
    model.finetuner.model_class.pretrained_config_filename=<config>
```

where

- `checkpoint` is the path to the pretrained model checkpoint, e.g. `epoch=9-step=39.ckpt`
- `dataset_state_dict` is the path to the dataset state dict of the pretrained model, e.g. `dataset_state_dict.yaml`
- `config` is the path to the config file of the pretrained model, e.g. `hydra_cfg_final.yaml`

To train for the `TPL100` (`Grambow`) dataset, replace `schneider.yaml` by
`tpl.yaml` (`grambow_green.yaml`) in datamodule.

### Direct supervised training

To train for the `Schneider` dataset:

```bash
python run.py datamodule=classification/schneider.yaml
```

For `TPL` (`Grambow`) dataset, set `datamodule` to `classification/tpl.yaml`
(`classification/grambow_green.yaml`).
