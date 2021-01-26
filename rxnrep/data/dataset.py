import functools
import logging
import multiprocessing
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import dgl
import torch

from rxnrep.core.molecule import Molecule
from rxnrep.core.reaction import Reaction
from rxnrep.data.grapher import (
    combine_graphs,
    create_hetero_molecule_graph,
    create_reaction_graph,
)
from rxnrep.data.transformer import HeteroGraphFeatureStandardScaler
from rxnrep.utils import convert_tensor_to_list, to_path, yaml_dump, yaml_load

logger = logging.getLogger(__name__)


class BaseDataset:
    """
    Base dataset class.
    """

    def __init__(
        self,
        reactions: List[Reaction],
        atom_featurizer: Callable,
        bond_featurizer: Callable,
        global_featurizer: Callable,
        *,
        init_state_dict: Optional[Union[Dict, Path]] = None,
        transform_features: bool = True,
        return_index: bool = True,
        num_processes: int = 1,
    ):

        self.reactions = reactions
        self.atom_featurizer = atom_featurizer
        self.bond_featurizer = bond_featurizer
        self.global_featurizer = global_featurizer
        self.init_state_dict = init_state_dict
        self.nprocs = num_processes
        self.return_index = return_index

        # will recover from state dict if it is not None
        self._species = None
        self._feature_mean = None
        self._feature_std = None
        self._label_mean = None
        self._label_std = None

        # recovery state info
        if init_state_dict is not None:
            # given as a dictionary
            if isinstance(init_state_dict, dict):
                self.load_state_dict(init_state_dict)
            # given as a file
            else:
                self.load_state_dict_file(init_state_dict)

        # convert reactions to dgl graphs (should do this after recovering state dict,
        # since the species, feature mean and stdev might be recovered there)
        self.dgl_graphs = self.build_graph_and_featurize()

        if transform_features:
            self.scale_features()

        # failed reactions
        self._failed = None

    @property
    def feature_size(self) -> Dict[str, int]:
        """
        Return the size of the features for each node type: {node_type, feature_size}.
        """
        size = {
            "atom": self.atom_featurizer.feature_size,
            "bond": self.bond_featurizer.feature_size,
            "global": self.global_featurizer.feature_size,
        }

        return size

    @property
    def feature_name(self) -> Dict[str, List[str]]:
        """
        Return the name of the features for each node type, {node_type, feature_name}.
        """
        name = {
            "atom": self.atom_featurizer.feature_name,
            "bond": self.bond_featurizer.feature_name,
            "global": self.global_featurizer.feature_name,
        }

        return name

    @property
    def label_mean(self) -> Dict[str, torch.Tensor]:
        return self._label_mean

    @property
    def label_std(self) -> Dict[str, torch.Tensor]:
        return self._label_std

    def get_failed(self) -> List[bool]:
        """
        Get the information of whether the reactions fails when converting them to graphs.

        The most prevalent failing reason is that cannot convert a smiles (pymatgen
        molecule graph) to a rdkit molecule.

        Returns:
            Each element indicates whether a reaction fails. The size of this list is the
            same as the number of reactions trying to read, each corresponding to a one
            reaction in the same order.
        """
        return self._failed

    def get_molecules(self) -> List[Molecule]:
        """
        Get all the molecules in the dataset.
        """
        molecules = []
        for rxn in self.reactions:
            molecules.extend(rxn.reactants + rxn.products)

        return molecules

    def get_molecule_graphs(self) -> List[dgl.DGLGraph]:
        """
        Get all the molecule graphs in the dataset.
        """
        graphs = []
        for reactants_g, products_g, _ in self.dgl_graphs:
            graphs.extend([reactants_g, products_g])

        return graphs

    def get_species(self) -> List[str]:
        """
        Get the species (atom types) appearing in all molecules in the dataset.
        """
        species = set()
        for mol in self.get_molecules():
            species.update(mol.species)

        return sorted(species)

    def get_charges(self) -> List[int]:
        """
        Get the (unique) charges of molecules.
        """
        charges = set([m.charge for m in self.get_molecules()])

        return sorted(charges)

    def build_graph_and_featurize(
        self,
    ) -> List[Tuple[dgl.DGLGraph, dgl.DGLGraph, dgl.DGLGraph]]:
        """
        Build DGL graphs for molecules in the reactions and then featurize the molecules.

        Each reaction is represented by three graphs, one for reactants, one for
        products, and the other for the union of the reactants and products.

        Returns:
            Each tuple represents on reaction, containing dgl graphs of the reactants,
            products, and their union.
        """

        logger.info("Starting building graphs and featurizing...")

        # self._species will not be None, if state_dict is provide. This will be the
        # case for retraining and test. If it is None, this is in the training mode,
        # and we get the species from the dataset.
        if self._species is None:
            self._species = self.get_species()

        atom_featurizer = functools.partial(
            self.atom_featurizer, allowable_atom_type=self._species
        )

        # build graph and featurize
        if self.nprocs == 1:
            dgl_graphs = [
                build_hetero_graph_and_featurize_one_reaction(
                    rxn,
                    atom_featurizer=atom_featurizer,
                    bond_featurizer=self.bond_featurizer,
                    global_featurizer=self.global_featurizer,
                    self_loop=True,
                )
                for rxn in self.reactions
            ]
        else:
            func = functools.partial(
                build_hetero_graph_and_featurize_one_reaction,
                atom_featurizer=atom_featurizer,
                bond_featurizer=self.bond_featurizer,
                global_featurizer=self.global_featurizer,
                self_loop=True,
            )
            with multiprocessing.Pool(self.nprocs) as p:
                dgl_graphs = p.map(func, self.reactions)

            # multiprocessing makes a copy of atom_featurizer and bond_featurizer and
            # then pass them to the subprocess. As a result, feature_name and
            # feature_size in the featurizer will not be updated.
            # Here we simply call it on the first reaction to initialize it
            build_hetero_graph_and_featurize_one_reaction(
                self.reactions[0],
                atom_featurizer=atom_featurizer,
                bond_featurizer=self.bond_featurizer,
                global_featurizer=self.global_featurizer,
                self_loop=True,
            )

        # log feature name and size
        for k in self.feature_name:
            ft_name = self.feature_name[k]
            ft_size = self.feature_size[k]
            logger.info(f"{k} feature name: {ft_name}")
            logger.info(f"{k} feature size: {ft_size}")

        logger.info("Finish building graphs and featurizing...")

        return dgl_graphs

    def scale_features(self):
        """
        Scale the feature values in the graphs by subtracting the mean and then
        dividing by standard deviation.
        """
        logger.info(f"Start scaling features...")

        # create new scaler
        if self.init_state_dict is None:
            feature_scaler = HeteroGraphFeatureStandardScaler(mean=None, std=None)

        # recover feature scaler mean and stdev
        else:
            assert (
                self._feature_mean is not None
            ), "Corrupted state_dict. Expect `feature_mean` to be a dict, got `None`."
            assert (
                self._feature_std is not None
            ), "Corrupted state_dict. Expect `feature_std` to be a dict, got `None`."

            feature_scaler = HeteroGraphFeatureStandardScaler(
                mean=self._feature_mean, std=self._feature_std
            )

        graphs = self.get_molecule_graphs()
        feature_scaler(graphs)  # graph features are updated inplace

        # save the mean and stdev of the feature scaler (should set after calling scaler)
        if self.init_state_dict is None:
            self._feature_mean = feature_scaler.mean
            self._feature_std = feature_scaler.std

        logger.info(f"Feature mean: {self._feature_mean}")
        logger.info(f"Feature std: {self._feature_std}")
        logger.info(f"Finish scaling features...")

    def scale_label(self, values: torch.Tensor, name: str) -> torch.Tensor:
        """
        Scale scalar labels.

        Args:
            values: 1D tensor of the labels
            name: name of the label

        Returns:
            1D tensor Scaled label values.
        """

        if self.init_state_dict is None:
            # compute from data
            mean = torch.mean(values)
            std = torch.std(values)

            if self._label_mean is None:
                self._label_mean = {name: mean}
            else:
                self._label_mean[name] = mean
            if self._label_std is None:
                self._label_std = {name: std}
            else:
                self._label_std[name] = std

        else:
            # recover from state dict
            assert (
                self.label_mean is not None
            ), "Corrupted state_dict. Expect `label_mean` to be a dict, got `None`."
            assert (
                self.label_std is not None
            ), "Corrupted state_dict. Expect `label_std` to be a dict, got `None`."
            mean = self.label_mean[name]
            std = self.label_std[name]

        values = (values - mean) / std

        logger.info(f"Label `{name}` mean: {mean}")
        logger.info(f"Label `{name}` std: {std}")

        return values

    def state_dict(self):
        d = {
            "species": self._species,
            "feature_name": self.feature_name,
            "feature_size": self.feature_size,
            "feature_mean": self._feature_mean,
            "feature_std": self._feature_std,
            "label_mean": self._label_mean,
            "label_std": self._label_std,
        }

        return d

    def load_state_dict(self, d: Dict[str, Any]):
        """
        Load state dict.

        Args:
            d: state dict
        """

        try:
            self._species = d["species"]
            self._feature_mean = d["feature_mean"]
            self._feature_std = d["feature_std"]
            self._label_mean = d["label_mean"]
            self._label_std = d["label_std"]

        except KeyError as e:
            raise ValueError(f"Corrupted state dict: {str(e)}")

        # sanity check: species should not be None
        assert (
            self._species is not None
        ), "Corrupted state_dict. Expect `species` to be a list, got `None`."

    def load_state_dict_file(self, filename: Path):
        """
        Load state dict from a yaml file.

        Args:
            filename: path of the file to load the data
        """

        def to_tensor(d: Dict[str, torch.Tensor], dtype: str = "float32"):
            dtype = getattr(torch, dtype)
            new_d = {k: torch.as_tensor(v, dtype=dtype) for k, v in d.items()}
            return new_d

        filename = to_path(filename)
        d = yaml_load(filename)

        try:
            species = d["species"]
            feature_mean = d["feature_mean"]
            feature_std = d["feature_std"]
            label_mean = d["label_mean"]
            label_std = d["label_std"]
            dtype = d["dtype"]

            # convert to tensors
            if feature_mean is not None and feature_std is not None:
                feature_mean = to_tensor(feature_mean, dtype)
                feature_std = to_tensor(feature_std, dtype)

            if label_mean is not None and label_std is not None:
                label_mean = to_tensor(label_mean, dtype)
                label_std = to_tensor(label_std, dtype)

            self._species = species
            self._feature_mean = feature_mean
            self._feature_std = feature_std
            self._label_mean = label_mean
            self._label_std = label_std

        except KeyError as e:
            raise ValueError(f"Corrupted state_dict (file): {str(e)}")

        # sanity check: species should not be None
        assert (
            self._species is not None
        ), "Corrupted state_dict file. Expect `species` to be a list, got `None`."

    def save_state_dict_file(self, filename: Optional[Union[str, Path]] = None):
        """
        Save the state dict to a yaml file.

        The data type of tensors are saved as a key `dtype`, which can be used in
        load_state_dict_file to convert the corresponding fields to tensor.

        Args:
            filename: path to save the file
        """

        def get_dtype(d: Dict[str, torch.Tensor]):
            key = list(d.keys())[0]
            dtype = d[key].dtype
            return dtype

        filename = self.init_state_dict if filename is None else filename
        filename = to_path(filename)

        # convert tensors to list if they exists
        tensor_fields = ["feature_mean", "feature_std", "label_mean", "label_std"]

        d = {}
        dtype = None
        for k, v in self.state_dict().items():
            if k in tensor_fields and v is not None:
                dtype = get_dtype(v)
                v = convert_tensor_to_list(v)
            d[k] = v

        # get a string representation of the later part of a dtype, e.g. torch.float32
        if dtype is not None:
            dtype = str(dtype).split(".")[1]

        d["dtype"] = dtype

        yaml_dump(d, filename)

    def __getitem__(self, item):
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.reactions)


class Subset(BaseDataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    @property
    def feature_size(self):
        return self.dataset.feature_size

    @property
    def feature_name(self):
        return self.dataset.feature_name

    def __getitem__(self, item):
        return self.dataset[self.indices[item]]

    def __len__(self):
        return len(self.indices)


def build_hetero_graph_and_featurize_one_reaction(
    reaction: Reaction,
    atom_featurizer: Callable,
    bond_featurizer: Callable,
    global_featurizer: Callable,
    self_loop=False,
) -> Tuple[dgl.DGLGraph, dgl.DGLGraph, dgl.DGLGraph]:
    """
    Build heterogeneous dgl graph for the reactants and products in a reaction and
    featurize them.

    Args:
        reaction:
        atom_featurizer:
        bond_featurizer:
        global_featurizer:
        self_loop:

    Returns:
        reactants_g: dgl graph for the reactants. One graph for all reactants; each
            disjoint subgraph for a molecule.
        products_g: dgl graph for the products. One graph for all reactants; each
            disjoint subgraph for a molecule.
        reaction_g: dgl graph for the reaction. bond nodes is the union of reactants
            bond nodes and products bond nodes.
    """

    def featurize_one_mol(m: Molecule):

        rdkit_mol = m.rdkit_mol
        # create graph
        g = create_hetero_molecule_graph(rdkit_mol, self_loop)

        # featurize molecules
        atom_feats = atom_featurizer(rdkit_mol)
        bond_feats = bond_featurizer(rdkit_mol)
        global_feats = global_featurizer(
            rdkit_mol, charge=m.charge, environment=m.environment
        )

        # add feats to graph
        g.nodes["atom"].data.update({"feat": atom_feats})
        g.nodes["bond"].data.update({"feat": bond_feats})
        g.nodes["global"].data.update({"feat": global_feats})

        return g

    try:
        reactant_graphs = [featurize_one_mol(m) for m in reaction.reactants]
        product_graphs = [featurize_one_mol(m) for m in reaction.products]

        # combine small graphs to form one big graph for reactants and products
        atom_map_number = reaction.get_reactants_atom_map_number(zero_based=True)
        bond_map_number = reaction.get_reactants_bond_map_number(for_changed=True)
        reactants_g = combine_graphs(reactant_graphs, atom_map_number, bond_map_number)

        atom_map_number = reaction.get_products_atom_map_number(zero_based=True)
        bond_map_number = reaction.get_products_bond_map_number(for_changed=True)
        products_g = combine_graphs(product_graphs, atom_map_number, bond_map_number)

        # combine reaction graph from the combined reactant graph and product graph
        num_unchanged = len(reaction.unchanged_bonds)
        num_lost = len(reaction.lost_bonds)
        num_added = len(reaction.added_bonds)

        reaction_g = create_reaction_graph(
            reactants_g,
            products_g,
            num_unchanged,
            num_lost,
            num_added,
            self_loop,
        )

    except Exception as e:
        logger.error(f"Error build graph and featurize for reaction: {reaction.id}")
        raise Exception(e)

    return reactants_g, products_g, reaction_g
