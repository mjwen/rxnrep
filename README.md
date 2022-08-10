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

### Training using mrnet reaction
First, update [electrolyte.yaml](./configs/datamodule/regression/electrolyte.yaml) 
by setting the `trainset_filename`, `valset_filename`, and `testset_filename` to the 
absolute paths to the data files on your machine. Example data files are located at 
[dataset/electrolyte](./dataset/electrolyte).

Next, train the model by 

```shell 
python run.py model/decoder=regressor.yaml datamodule=regression/electrolyte.yaml
```

### Training using SMILES reaction 

Instead of using mrnet reaction, reactions can be provided as SMILES strings. See 
 [dataset/nrel](./dataset/nrel) for example datasets. 

To train, first update [nrel.yaml](./configs/datamodule/regression/nrel.yaml)  
by setting the `trainset_filename`, `valset_filename`, and `testset_filename` to the 
absolute paths to the data files on your machine (e.g. on in [dataset/nrel](./dataset/nrel)). 

Then, train the model by 

```shell
python run.py model/decoder=regressor.yaml datamodule=regression/nrel.yaml
```


## Use BonDNet for prediction 

### Use pretrained model for prediction
```shell
bondnet <path_to_data_file> 
```

where `path_to_data_file` is a json file containing a list of mrnet 
[Reaction](https://github.com/materialsproject/mrnet/blob/84f4814a565753060d81cf18ab48e8f71fff6fd8/src/mrnet/core/reactions.py#L46) 
you want to predict; of course, subclasses of 
[Reaction](https://github.com/materialsproject/mrnet/blob/84f4814a565753060d81cf18ab48e8f71fff6fd8/src/mrnet/core/reactions.py#L46)
is OK.
An example data file can be found at [examples/reactions_mrnet.json](./examples/). 
Note, depending on how the mrnet reactions are constructed, atom mapping may or may not 
be there. If not, you can use 
[get_reaction_atom_mapping()](https://github.com/materialsproject/mrnet/blob/84f4814a565753060d81cf18ab48e8f71fff6fd8/src/mrnet/utils/reaction.py#L25)
to generate the atom mappings.

Running this command will generate a file named `results.json`, which is a copy of the 
input data file, but with an additional key `predicted_reaction_energy` inserted to each 
entry, the value of which gives the BDE if the reaction is a one-bond breaking one. 

### Use your own model for prediction

If you've trained your own model and want to use it for prediction, you can do

```shell
bondnet <path_to_data_file>  --model <path_to_model>
```

Again, `path_to_data_file` is a file containing the reactions you want to make 
predictions for. 
`path_to_model` is the path to the model you've trained. More specifically, 
after training, models will be saved to a directories looking like 
```plain
- outputs
  ... 
  - 2022-08-09
    - 14-22-51
    - 16-44-02
  - 2022-08-10
    - 10-10-02
    - 15-24-01
```
The latest model will always be the last one in the list. So, for example, you can 
then set `<path_to_model>` to `./outputs/2022-08-10/15-24-01` to use the latest model 
for 
prediction.
