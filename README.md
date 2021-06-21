# RxnRep (Reaction Representation): Mapping the atlas of chemical reactions


## Installation 

```shell
git clone https://github.com/mjwen/rxnrep.git
cd rxnrep
git checkout bondnet 
conda env create -f environment.yml
conda activate rxnrep
pip install -e . 
```

## To train BonDNet

```shell 
python run.py model/decoder=regressor.yaml datamodule=regression/electrolyte.yaml
```


## Use BonDNet for prediction 

```shell
bondnet <path_to_data_file> 
```

where `path_to_data_file` is a json file containing mrnet reactions. This will 
generate a filenamed `results.json`, which is a copy of the input file, but with an 
additional key `predicted_reaction_energy` inserted for each entry. The 
`predicted_reactoin_energy` gives the bond dissociation energy if the reaction is 
a one-bond breaking reaction.
