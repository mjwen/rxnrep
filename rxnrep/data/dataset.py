import logging
import dgl
from pathlib import Path
from typing import List, Callable, Tuple, Optional, Dict, Any
from rxnrep.core.molecule import Molecule
from rxnrep.core.reaction import Reaction
from rxnrep.data.transformer import HeteroGraphFeatureStandardScaler

logger = logging.getLogger(__name__)


class BaseDataset:
    """
    Base dataset class.
    """

    def __init__(
        self,
        reactions: List[Reaction],
        labels: List[List[int]],
        atom_featurizer: Callable,
        bond_featurizer: Callable,
        global_featurizer: Callable,
        state_dict_filename: Optional[str, Path] = None,
        num_processes: int = 1,
    ):

        self.raw_reactions = reactions
        self.raw_labels = labels
        self.atom_featurizer = atom_featurizer
        self.bond_featurizer = bond_featurizer
        self.global_featurizer = global_featurizer
        self.state_dict_filename = state_dict_filename
        self.nprocs = num_processes

        self.reactions = None
        self.labels = None

        self._feature_size = None
        self._feature_name = None
        self._feature_scaler_mean = None
        self._feature_scaler_std = None

        self._species = None
        self._failed = None

    @property
    def feature_size(self) -> Dict[str, int]:
        """
        Return the size of the features for each node type: {node_type, feature_size}.
        """
        return self._feature_size

    @property
    def feature_name(self) -> Dict[str, List[str]]:
        """
        Return the name of the features for each node type, {node_type, feature_name}.
        """
        return self._feature_name

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
        for rxn in self.raw_reactions:
            molecules.extend(rxn.reactants + rxn.products)

        return molecules

    # def get_rdkit_molecules(self) -> List[Chem.Mol]:
    #     """
    #     Get all the molecules (rdkit molecules) in the dataset.
    #     """
    #     molecules = self.get_molecules()
    #     molecules = [m.rdkit_mol for m in molecules]
    #
    #     return molecules

    def get_graphs(self) -> List[dgl.DGLGraph]:
        """
        Get all the graphs in the dataset.
        """
        graphs = []
        for reactants, products in self.reactions:
            graphs.extend([reactants, products])

        return graphs

    def get_species(self) -> List[str]:
        """
        Get the species (atom types) appearing in all molecules in the dataset.
        """
        species = set()
        for mol in self.get_molecules():
            species.update(mol.species)

        return sorted(species)

    def scale_features(self):
        """
        Scale the feature values in the graphs by subtracting the mean and then
        dividing by standard deviation.
        """
        logger.info(f"Start scaling features...")

        # create new scaler
        if self.state_dict_filename is None:
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

        graphs = self.get_graphs()
        feature_scaler(graphs)  # graph features are updated inplace

        # save the mean and stdev of the feature scaler
        if self.state_dict_filename is None:
            self._feature_scaler_mean = feature_scaler.mean
            self._feature_scaler_std = feature_scaler.std

        logger.info(f"Feature scaler mean: {self._feature_scaler_mean}")
        logger.info(f"Feature scaler std: {self._feature_scaler_std}")
        logger.info(f"Finish scaling features...")

    def load_state_dict(self, d: Dict[str, Any]):
        try:
            self._species = d["species"]
            self._feature_size = d["feature_size"]
            self._feature_name = d["feature_name"]
            self._feature_scaler_mean = d["feature_scaler_mean"]
            self._feature_scaler_std = d["feature_scaler_std"]
        except KeyError as e:
            raise ValueError(f"Corrupted state_dict (file): {str(e)}")

        # sanity check: species, feature size, and feature name should not be None
        assert (
            self._species is not None
        ), "Corrupted state_dict file. Expect `species` to be a list, got `None`."
        assert (
            self._feature_size is not None
        ), "Corrupted state_dict file. Expect `feature_size` to be a dict, got `None`."
        assert (
            self._feature_name is not None
        ), "Corrupted state_dict file. Expect `feature_name` to be a dict got `None`."

    def state_dict(self):
        d = {
            "species": self._species,
            "feature_size": self._feature_size,
            "feature_name": self._feature_name,
            "feature_scaler_mean": self._feature_scaler_mean,
            "feature_scaler_std": self._feature_scaler_std,
        }

        return d

    def __getitem__(self, item: int) -> Tuple[dgl.DGLGraph, dgl.DGLGraph, Any]:
        """Get data point with index.

        Args:
            item: data point index

        Returns:
            reactants: reactants of the reactions
            products: products of the reactions
            label: label for the reaction
        """
        reactants, products = self.reactions[item]
        label = self.labels[item]

        return reactants, products, label

    def __len__(self) -> int:
        """
        Returns length of dataset (i.e. number of reactions)
        """
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
