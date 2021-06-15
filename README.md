# RxnRep 

Contrastive pretraining of chemical reaction GNN, boosting performance of chemical 
reaction classification for small dataset.


# Installation 

```bash
conda env create -f environment.yml
conda activate rxnrep
pip install -e . 
```


# Train the model

### Pretraining & fine-tuning 

To pretrain the model using contrastive learning:

```bash
python run.py --config-name config_contrastive.yaml datamodule=contrastive/schneider.yaml 
```

To finetune the pretrained model:

```bash
python run.py --config-name config_finetune.yaml datamodule=classification/schneider.yaml pretrained_wandb_id=<wandb_id>
```
where `wandb_id` is the W&B id for the pretraining run, an eight-character alphanumeric 
e.g. `3oep187z`. By providing the `wandb_id`, the finetuning script will search for the 
necessary pretrained model info automatically.

Alternatively, the pretrained model info can be passed in manually: 

```bash
python run.py --config-name config_finetune.yaml datamodule=classification/schneider.yaml \
    model.finetuner.model_class.pretrained_checkpoint_filename=<checkpoint> \
    model.finetuner.model_class.pretrained_dataset_state_dict_filename=<dataset_state_dict> \ 
    model.finetuner.model_class.pretrained_config_filename=<config>
```
where 
- `checkpoint` is path to the pretrained model checkpoint, e.g. `epoch=9-step=39.ckpt`;
- `dataset_state_dict` is path to the dataset state dict of the pretrained model, e.g. `dataset_state_dict.yaml`;
- `config` is path to the config file of the pretrained model, e.g. `hydra_cfg_final.yaml`


To train for the `TPL` (`Grambow-Green`) dataset, replace `schneider.yaml` by 
`tpl.yaml` (`grambow_green.yaml`) in datamodule.



### Direct supervised training 

To train for the `Schneider` dataset:

```bash
python run.py datamodule=classification/schneider.yaml 
```
  
For `TPL` (`Grambow-Green`) dataset, set `datamodule` to `classification/tpl.yaml` 
(`classification/grambow_green.yaml`).