# RxnRep

Self-supervised contrastive pretraining for chemical reaction representation ([RxnRep](https://doi.org/10.1039/d1sc06515g)).

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

## Get RxnRep reaction fingerprints

To convert SMILES reactions to RxnRep fingerprints, simply do something like the below:

```python
from rxnrep.predict.fingerprint import get_rxnrep_fingerprint

rxn1 = "[CH3:6][CH2:7][OH:16].[O:1]=[C:2]([C:3](=[O:4])[OH:5])[CH2:8][CH2:9][c:10]1[cH:11][cH:12][cH:13][cH:14][cH:15]1>>[O:1]=[C:2]([C:3](=[O:4])[O:5][CH2:6][CH3:7])[CH2:8][CH2:9][c:10]1[cH:11][cH:12][cH:13][cH:14][cH:15]1.[OH2:16]"
rxn2 = "[C:1](#[N:2])[c:3]1[cH:4][cH:5][c:6]([CH2:7][C:8](=[O:9])[OH:10])[cH:13][cH:14]1.[CH3:11][CH2:12][OH:15]>>[C:1](#[N:2])[c:3]1[cH:4][cH:5][c:6]([CH2:7][C:8](=[O:9])[O:10][CH2:11][CH3:12])[cH:13][cH:14]1.[OH2:15]"

smiles_reactions = [rxn1, rxn2]
fingerprints = get_rxnrep_fingerprint(smiles_reactions)

print(fingerprints.shape)  # torch.size([2, 128])
```

See the docs of `get_rxnrep_fingerprint()` for more options, e.g. choosing which
pretrained model to use and fine-tuning the fingerprints.

## Train classification models

### Direct supervised training

To train for the `Schneider` dataset:

```bash
python run.py --config-name config.yaml  datamodule=classification/schneider.yaml
```

For `TPL100` (`Grambow`) dataset, set `datamodule` to `classification/tpl100.yaml`
(`classification/grambow.yaml`).

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

where `wandb_id` is the [W&B](https://wandb.ai) id for the pretraining run, an eight-character
alphanumeric
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
`tpl100.yaml` (`grambow.yaml`) in datamodule.

## Train regression models

Train regression models are very similar to what discussed above for classification
models. See [here](./configs/README.md) for detailed info.

## Config the training

The training are configured using [hydra](https://github.com/facebookresearch/hydra)
and the configuration files are at [configs](./configs).

## Cite

```
@article{wen2022rxnrep,
  title   = {Improving machine learning performance on small chemical reaction data
  with unsupervised contrastive pretraining},
  author  = {Wen, Mingjian and Blau, Samuel M and Xie, Xiaowei and Dwaraknath, Shyam
  and Persson, Kristin A},
  journal = {Chemical Science},
  year    = 2022,
  doi     = {10.1039/D1SC06515G},
  url     = {https://doi.org/10.1039/D1SC06515G},
}
```
