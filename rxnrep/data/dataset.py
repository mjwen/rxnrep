import logging
import dgl
import torch
from pathlib import Path
from typing import List, Callable, Optional, Dict, Any, Union
from rxnrep.core.molecule import Molecule
from rxnrep.core.reaction import Reaction
from rxnrep.data.transformer import HeteroGraphFeatureStandardScaler
from rxnrep.utils import to_path

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
        state_dict_filename: Optional[Union[str, Path]] = None,
        num_processes: int = 1,
    ):

        self.reactions = reactions
        self.atom_featurizer = atom_featurizer
        self.bond_featurizer = bond_featurizer
        self.global_featurizer = global_featurizer
        self.state_dict_filename = state_dict_filename
        self.nprocs = num_processes

        self._species = None
        self._feature_scaler_mean = None
        self._feature_scaler_std = None

        self._failed = None

        # recovery state info
        if state_dict_filename is not None:
            state_dict_filename = torch.load(str(to_path(state_dict_filename)))
            self.load_state_dict(state_dict_filename)

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

    # def get_rdkit_molecules(self) -> List[Chem.Mol]:
    #     """
    #     Get all the molecules (rdkit molecules) in the dataset.
    #     """
    #     molecules = self.get_molecules()
    #     molecules = [m.rdkit_mol for m in molecules]
    #
    #     return molecules

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

        graphs = self.get_molecule_graphs()
        feature_scaler(graphs)  # graph features are updated inplace

        # save the mean and stdev of the feature scaler (should set after calling scaler)
        if self.state_dict_filename is None:
            self._feature_scaler_mean = feature_scaler.mean
            self._feature_scaler_std = feature_scaler.std

        logger.info(f"Feature scaler mean: {self._feature_scaler_mean}")
        logger.info(f"Feature scaler std: {self._feature_scaler_std}")
        logger.info(f"Finish scaling features...")

    def load_state_dict(self, d: Dict[str, Any]):
        try:
            self._species = d["species"]
            self._feature_scaler_mean = d["feature_scaler_mean"]
            self._feature_scaler_std = d["feature_scaler_std"]
        except KeyError as e:
            raise ValueError(f"Corrupted state_dict (file): {str(e)}")

        # sanity check: species, feature size, and feature name should not be None
        assert (
            self._species is not None
        ), "Corrupted state_dict file. Expect `species` to be a list, got `None`."

    def state_dict(self):
        d = {
            "species": self._species,
            "feature_scaler_mean": self._feature_scaler_mean,
            "feature_scaler_std": self._feature_scaler_std,
        }

        return d


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