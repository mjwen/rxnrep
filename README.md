# RxnRep (Reaction Representation): Mapping the atlas of chemical reactions


## Installation 

```shell
git clone https://github.com/mjwen/rxnrep.git
cd rxnrep
git checkout bondnet 
conda env create -f environment.yml
conda activate rxnrep
pip install -r requirements.txt
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

where `path_to_data_file` is a json file containing a list of mrnet 
[Reaction](https://github.com/materialsproject/mrnet/blob/84f4814a565753060d81cf18ab48e8f71fff6fd8/src/mrnet/core/reactions.py#L46) 
you want to predict; of course, subclasses of 
[Reaction](https://github.com/materialsproject/mrnet/blob/84f4814a565753060d81cf18ab48e8f71fff6fd8/src/mrnet/core/reactions.py#L46)
is OK.
An example data file can be found at `examples/reactions_mrnet.json`. 
Note, depending on how the mrnet reactions are constructed, atom mapping may or may not 
be there. If not, you can use 
[get_reaction_atom_mapping()](https://github.com/materialsproject/mrnet/blob/84f4814a565753060d81cf18ab48e8f71fff6fd8/src/mrnet/utils/reaction.py#L25)
to generate the atom mappings.



Running this command will generate a file named `results.json`, which is a copy of the 
input data file, but with an additional key `predicted_reaction_energy` inserted to each 
entry, the value of which gives the BDE if the reaction is a one-bond breaking one. 
