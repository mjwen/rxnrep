import logging
import dgl
import torch
from pathlib import Path
from typing import List, Callable, Optional, Dict, Any, Union
from rxnrep.core.molecule import Molecule
from rxnrep.core.reaction import Reaction
from rxnrep.data.transformer import HeteroGraphFeatureStandardScaler
from rxnrep.utils import to_path, yaml_dump, yaml_load, convert_tensor_to_list

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
        init_state_dict: Optional[Union[Dict, Path]] = None,
        num_processes: int = 1,
        return_index: bool = True,
    ):

        self.reactions = reactions
        self.atom_featurizer = atom_featurizer
        self.bond_featurizer = bond_featurizer
        self.global_featurizer = global_featurizer
        self.init_state_dict = init_state_dict
        self.nprocs = num_processes
        self.return_index = return_index

        self._species = None
        self._feature_scaler_mean = None
        self._feature_scaler_std = None

        self._failed = None

        # recovery state info
        if init_state_dict is not None:
            # given as a dictionary
            if isinstance(init_state_dict, dict):
                self.load_state_dict(init_state_dict)
            # given as a file
            else:
                self.load_state_dict_file(init_state_dict)

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
        raise NotImplementedError

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
            assert self._feature_scaler_mean is not None, (
                "Corrupted state_dict file. Expect `feature_scaler_mean` to be a list, "
                "got `None`."
            )
            assert (
                self._feature_scaler_std is not None
            ), "Corrupted state_dict file. Expect `feature_scaler_std` to be a list, "
            "got `None`."

            feature_scaler = HeteroGraphFeatureStandardScaler(
                mean=self._feature_scaler_mean, std=self._feature_scaler_std
            )

        graphs = self.get_molecule_graphs()
        feature_scaler(graphs)  # graph features are updated inplace

        # save the mean and stdev of the feature scaler (should set after calling scaler)
        if self.init_state_dict is None:
            self._feature_scaler_mean = feature_scaler.mean
            self._feature_scaler_std = feature_scaler.std

        logger.info(f"Feature scaler mean: {self._feature_scaler_mean}")
        logger.info(f"Feature scaler std: {self._feature_scaler_std}")
        logger.info(f"Finish scaling features...")

    def state_dict(self):
        d = {
            "species": self._species,
            "feature_name": self.feature_name,
            "feature_size": self.feature_size,
            "feature_scaler_mean": self._feature_scaler_mean,
            "feature_scaler_std": self._feature_scaler_std,
        }

        return d

    def load_state_dict(self, d: Optional[Dict] = None):
        """
        Load state dict from a yaml file.

        Args:
            d: state dict
        """
        d = self.init_state_dict if d is None else d

        try:
            species = d["species"]
            scaler_mean = d["feature_scaler_mean"]
            scaler_std = d["feature_scaler_std"]
            self._species = species
            self._feature_scaler_mean = scaler_mean
            self._feature_scaler_std = scaler_std

        except KeyError as e:
            raise ValueError(f"Corrupted state dict: {str(e)}")

        # sanity check: species should not be None
        assert (
            self._species is not None
        ), "Corrupted state_dict. Expect `species` to be a list, got `None`."

    def load_state_dict_file(self, filename: Optional[Union[str, Path]] = None):
        """
        Load state dict from a yaml file.

        Args:
            filename: path of the file to load the data
        """

        def to_tensor(d: Dict[str, torch.Tensor], dtype: str = "float32"):
            dtype = getattr(torch, dtype)
            new_d = {k: torch.as_tensor(v, dtype=dtype) for k, v in d.items()}
            return new_d

        filename = self.init_state_dict if filename is None else filename
        filename = to_path(filename)
        d = yaml_load(filename)

        try:
            species = d["species"]
            scaler_mean = d["feature_scaler_mean"]
            scaler_std = d["feature_scaler_std"]
            dtype = d["dtype"]

            # convert tensors
            if dtype is not None:
                scaler_mean = to_tensor(scaler_mean, dtype)
                scaler_std = to_tensor(scaler_std, dtype)

            self._species = species
            self._feature_scaler_mean = scaler_mean
            self._feature_scaler_std = scaler_std

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
        tensor_fields = ["feature_scaler_mean", "feature_scaler_std"]
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
