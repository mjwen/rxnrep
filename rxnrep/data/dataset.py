import copy
import functools
import logging
import multiprocessing
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import dgl
import numpy as np
import torch
from rdkit import Chem

from rxnrep.core.molecule import Molecule
from rxnrep.core.reaction import (
    Reaction,
    get_atom_bond_hop_dist_class_weight,
    get_atom_distance_to_reaction_center,
    get_bond_distance_to_reaction_center,
)
from rxnrep.data.augmentation import AtomTypeFeatureMasker
from rxnrep.data.scaler import GraphFeatureScaler, StandardScaler1D
from rxnrep.data.to_graph import build_graph_and_featurize_reaction
from rxnrep.data.transforms import MaskAtomAttribute, MaskBondAttribute
from rxnrep.utils import tensor_to_list, to_path, to_tensor, yaml_dump, yaml_load

logger = logging.getLogger(__name__)


class BaseDataset:
    """
    Base dataset.

    This base dataset deal with input molecule/reaction graphs and their features.

    Labels are added in subclass dataset.

    Args:
        filename: path to the dataset file
        atom_featurizer: function to create atom features
        bond_featurizer: function to create bond features
        global_featurizer: function to create global features
        build_reaction_graph: whether to create reaction graph.
        init_state_dict: initial state dict (or a yaml file of the state dict) containing
            the state of the dataset used for training: including all the atom types in
            the molecules, mean and stdev of the features (if transform_features is
            `True`). If `None`, these properties are computed from the current dataset.
        transform_features: whether to standardize the atom, bond, and global features.
            If `True`, each feature column will first subtract the mean and then divide
            by the standard deviation.
        return_index: whether to return the index of the sample in the dataset
        num_processes: number of processes used to load and process the dataset.
    """

    def __init__(
        self,
        filename: Union[str, Path],
        atom_featurizer: Callable[[Chem.Mol], torch.Tensor],
        bond_featurizer: Callable[[Chem.Mol], torch.Tensor],
        global_featurizer: Callable[[Chem.Mol], torch.Tensor],
        *,
        build_reaction_graph: bool = True,
        init_state_dict: Optional[Union[Dict, Path]] = None,
        transform_features: bool = True,
        return_index: bool = True,
        num_processes: int = 1,
    ):
        self.atom_featurizer = atom_featurizer
        self.bond_featurizer = bond_featurizer
        self.global_featurizer = global_featurizer
        self.build_reaction_graph = build_reaction_graph
        self.init_state_dict = init_state_dict
        self.return_index = return_index
        self.nprocs = num_processes

        # read input files
        self.reactions, self._failed = self.read_file(filename)

        # ===== build graph and features =====
        # will recover from state dict if it is not None
        self._species = None
        self._feature_scaler_state_dict = None
        self._label_scaler = None
        self._label_scaler_state_dict = None

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

    def get_label_scaler(self) -> Dict[str, StandardScaler1D]:
        """
        Return the transformer used to scale the labels.

        Returns:
            {property: scaler}, where property is the name of the label to which the
                transformer is applied.
        """
        return self._label_scaler

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
                build_graph_and_featurize_reaction(
                    rxn,
                    atom_featurizer=atom_featurizer,
                    bond_featurizer=self.bond_featurizer,
                    global_featurizer=self.global_featurizer,
                    build_reaction_graph=self.build_reaction_graph,
                )
                for rxn in self.reactions
            ]
        else:
            func = functools.partial(
                build_graph_and_featurize_reaction,
                atom_featurizer=atom_featurizer,
                bond_featurizer=self.bond_featurizer,
                global_featurizer=self.global_featurizer,
                need_reaction_graph=self.build_reaction_graph,
            )
            with multiprocessing.Pool(self.nprocs) as p:
                dgl_graphs = p.map(func, self.reactions)

            # multiprocessing makes a copy of atom_featurizer and bond_featurizer and
            # then pass them to the subprocess. As a result, feature_name and
            # feature_size in the featurizer will not be updated.
            # Here we simply call it on the first reaction to initialize it
            build_graph_and_featurize_reaction(
                self.reactions[0],
                atom_featurizer=atom_featurizer,
                bond_featurizer=self.bond_featurizer,
                global_featurizer=self.global_featurizer,
                build_reaction_graph=self.build_reaction_graph,
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

        feature_scaler = GraphFeatureScaler()

        if self.init_state_dict is not None:
            assert (
                self._feature_scaler_state_dict is not None
            ), "Corrupted state_dict. `feature_scaler_state_dict is `None`."
            feature_scaler.load_state_dict(self._feature_scaler_state_dict)

        graphs = self.get_molecule_graphs()
        # graph features are updated inplace
        feature_scaler.transform(graphs)

        # keep it to saved to dataset state dict
        self._feature_scaler_state_dict = feature_scaler.state_dict()

        logger.info(f"Feature mean: {self._feature_scaler_state_dict['mean']}")
        logger.info(f"Feature std: {self._feature_scaler_state_dict['std']}")
        logger.info(f"Finish scaling features...")

    def state_dict(self):
        if self._label_scaler is not None:
            self._label_scaler_state_dict = {
                name: scaler.state_dict() for name, scaler in self._label_scaler.items()
            }

        d = {
            "species": self._species,
            "feature_name": self.feature_name,
            "feature_size": self.feature_size,
            "feature_scaler_state_dict": self._feature_scaler_state_dict,
            "label_scaler_state_dict": self._label_scaler_state_dict,
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
            self._feature_scaler_state_dict = d["feature_scaler_state_dict"]
            self._label_scaler_state_dict = d["label_scaler_state_dict"]

        except KeyError as e:
            raise ValueError(f"Corrupted state dict: {str(e)}")

        # sanity check: species should not be None
        assert (
            self._species is not None
        ), "Corrupted state_dict. Expect `species` to be a list, got `None`."

    def save_state_dict_file(self, filename: Optional[Union[str, Path]] = None):
        """
        Save the state dict to a yaml file.

        The data type of tensors are saved as a key `dtype`, which can be used in
        load_state_dict_file to convert the corresponding fields to tensor.

        Args:
            filename: path to save the file
        """

        filename = self.init_state_dict if filename is None else filename
        filename = to_path(filename)

        # convert tensors to list if they exists
        tensor_fields = ["feature_scaler_state_dict", "label_scaler_state_dict"]

        d = {}
        for k, v in self.state_dict().items():
            if k in tensor_fields:
                v = tensor_to_list(v)
            d[k] = v

        yaml_dump(d, filename)

    def load_state_dict_file(self, filename: Path):
        """
        Load state dict from a yaml file.

        Args:
            filename: path of the file to load the data
        """

        filename = to_path(filename)
        d = yaml_load(filename)

        try:
            species = d["species"]
            feature_scaler_state_dict = d["feature_scaler_state_dict"]
            label_scaler_state_dict = d["label_scaler_state_dict"]

        except KeyError as e:
            raise ValueError(f"Corrupted state_dict (file): {str(e)}")

        # sanity check: species should not be None
        assert (
            species is not None
        ), "Corrupted state_dict file. Expect `species` to be a list, got `None`."

        self._species = species
        self._feature_scaler_state_dict = to_tensor(feature_scaler_state_dict)
        self._label_scaler_state_dict = to_tensor(label_scaler_state_dict)

    def read_file(self, filename: Path) -> Tuple[List[Reaction], List[bool]]:
        """
        Read reactions from dataset file.

        Args:
            filename: name of the dataset

        Returns:
            reactions: a list of rxnrep Reaction succeed in converting to dgl graphs.
                The length of this list could be shorter than the number of entries in
                the dataset file (when some entry fails).
            failed: a list of bool indicating whether each entry in the dataset file
                fails or not. The length of the list is the same as the number of
                entries in the dataset file.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.reactions)

    def __getitem__(self, item):
        raise NotImplementedError


class BaseDatasetWithLabels(BaseDataset):
    """
    Base dataset with three types of labels:
    atom_hop_distance, bond_hop_distance, atom_type_masker

    Args:
        max_hop_distance: maximum allowed hop distance from the reaction center for
            atom and bond. Used to determine atom and bond label. If `None`, not used.
        atom_type_masker_ratio: ratio of atoms whose atom type to be masked in each
            reaction. If `None`, not applied.
        atom_type_masker_use_masker_value: whether to use the calculate masker value. If
            `True`, use it, if `False` do not mask the atom features.
            Ignored if `atom_type_masker_ratio` is None.
            Note the difference between this and `atom_type_masker_ratio`; if
            `atom_type_masker_ratio` is `None`, the masker is not used in the sense
            that the masker is not even created. But `atom_type_masker_use_mask_value`
            be of `False` means that the masker are created, but we simply skip the
            change of the atom type features of the masked atoms.
        allow_label_scaler_none: when `init_state_dict` is provided, whether to allow
            label_scaler is None in state dict. If `False`, will use the label scaler
            mean and std. If `True`, will recompute label mean and std. This is mainly
            used for the pretrain-finetune case, where the pretrain model does not use
            labels (e.g. contrastive training).
    """

    def __init__(
        self,
        filename: Union[str, Path],
        atom_featurizer: Callable,
        bond_featurizer: Callable,
        global_featurizer: Callable,
        *,
        build_reaction_graph=True,
        init_state_dict: Optional[Union[Dict, Path]] = None,
        transform_features: bool = True,
        return_index: bool = True,
        num_processes: int = 1,
        #
        # args to control labels
        #
        allow_label_scaler_none: bool = False,
        max_hop_distance: Optional[int] = None,
        atom_type_masker_ratio: Optional[float] = None,
        atom_type_masker_use_masker_value: Optional[bool] = None,
    ):

        super().__init__(
            filename,
            atom_featurizer,
            bond_featurizer,
            global_featurizer,
            build_reaction_graph=build_reaction_graph,
            init_state_dict=init_state_dict,
            transform_features=transform_features,
            return_index=return_index,
            num_processes=num_processes,
        )

        # labels and metadata (one inner dict for each reaction)
        # do not use [{}] * len(self.reactions); update one will change all
        self.labels = [{} for _ in range(len(self.reactions))]
        self.medadata = [{} for _ in range(len(self.reactions))]

        self.allow_label_scaler_none = allow_label_scaler_none

        # atom bond hop distance label
        self.max_hop_distance = max_hop_distance

        # atom type maker label (this need special handling because we generate it
        # dynamically; different label is generated each time it is called)
        if atom_type_masker_ratio is None:
            self.atom_type_masker = None
        else:
            self.atom_type_masker = AtomTypeFeatureMasker(
                allowable_types=self._species,
                feature_name=self.feature_name["atom"],
                feature_mean=self._feature_scaler_state_dict["mean"]["node"]["atom"],
                feature_std=self._feature_scaler_state_dict["mean"]["node"]["atom"],
                ratio=atom_type_masker_ratio,
                use_masker_value=atom_type_masker_use_masker_value,
            )

            # move the storage of atom features from the graph to self.atom_features
            self.atom_features = [
                {
                    "reactants": reactants_g.nodes["atom"].data.pop("feat"),
                    "products": products_g.nodes["atom"].data.pop("feat"),
                }
                for reactants_g, products_g, _ in self.dgl_graphs
            ]

        self.generate_labels()
        self.generate_metadata()

    def generate_labels(self):

        for i, rxn in enumerate(self.reactions):

            # atom bond hop distance label
            if self.max_hop_distance is not None:

                atom_hop = get_atom_distance_to_reaction_center(
                    rxn, max_hop=self.max_hop_distance
                )

                bond_hop = get_bond_distance_to_reaction_center(
                    rxn, atom_hop_distances=atom_hop, max_hop=self.max_hop_distance
                )
                self.labels[i].update(
                    {
                        "atom_hop_dist": torch.as_tensor(atom_hop, dtype=torch.int64),
                        "bond_hop_dist": torch.as_tensor(bond_hop, dtype=torch.int64),
                    }
                )

    def generate_metadata(self):

        for i, (rxn, label) in enumerate(zip(self.reactions, self.labels)):
            num_unchanged = len(rxn.unchanged_bonds)
            meta = {
                "reactant_num_molecules": len(rxn.reactants),
                "product_num_molecules": len(rxn.products),
                "num_atoms": rxn.num_atoms,
                "num_bonds": num_unchanged + len(rxn.lost_bonds) + len(rxn.added_bonds),
                "num_unchanged_bonds": num_unchanged,
                "num_reactant_bonds": num_unchanged + len(rxn.lost_bonds),
                "num_product_bonds": num_unchanged + len(rxn.added_bonds),
                "atoms_in_reaction_center": (
                    np.asarray(rxn.atom_distance_to_reaction_center) == 0
                ).tolist(),
                "bonds_in_reaction_center": (
                    np.asarray(rxn.bond_distance_to_reaction_center) == 0
                ).tolist(),
            }

            # add atom/bond hop dist to meta, which is used in hop dist pool
            if self.max_hop_distance is not None:
                meta.update(
                    {
                        "atom_hop_dist": label["atom_hop_dist"],
                        "bond_hop_dist": label["bond_hop_dist"],
                    }
                )

            self.medadata[i].update(meta)

    def scale_label(self, values: torch.Tensor, name: str) -> torch.Tensor:
        """
        Scale scalar labels.

        Args:
            values: 1D tensor of the labels
            name: name of the label

        Returns:
            1D tensor Scaled label values.
        """
        assert (
            len(values.shape) == 1
        ), f"Expect 1D tensor as input; got {len(values.shape)}."

        label_scaler = StandardScaler1D()

        if self.init_state_dict is not None and not self.allow_label_scaler_none:
            assert self._label_scaler_state_dict is not None, (
                "Corrupted state_dict. Expect `label_scaler_state_dict` to be a dict, "
                "got `None`."
            )

            label_scaler.load_state_dict(self._label_scaler_state_dict[name])

        values = label_scaler.transform(values)

        if self._label_scaler is None:
            self._label_scaler = {}
        self._label_scaler[name] = label_scaler

        state_dict = label_scaler.state_dict()
        logger.info(f"Label `{name}` mean: {state_dict['mean']}")
        logger.info(f"Label `{name}` std: {state_dict['std']}")

        return values

    def get_class_weight(self, only_break_bond: bool = False):
        """
        Create class weight to be used in cross entropy losses.

        Args:
            only_break_bond: whether the dataset only contains breaking bond, i.e.
                does not have lost bond
        """
        if self.max_hop_distance:
            return get_atom_bond_hop_dist_class_weight(
                self.labels, self.max_hop_distance, only_break_bond=only_break_bond
            )
        else:
            return {}

    def __getitem__(self, item: int):
        """
        Get data point with index.
        """
        reactants_g, products_g, reaction_g = self.dgl_graphs[item]
        reaction = self.reactions[item]
        label = self.labels[item]
        meta = self.medadata[item]

        # atom type masker (this needs to be here since we generate it dynamically;
        # different label is generated each time it is called)
        if self.atom_type_masker is not None:
            atom_feats = self.atom_features[item]

            # Assign atom features bach to graph. Should clone to keep
            # self.atom_features intact.
            reactants_g.nodes["atom"].data["feat"] = (
                atom_feats["reactants"].clone().detach()
            )
            products_g.nodes["atom"].data["feat"] = (
                atom_feats["products"].clone().detach()
            )

            (
                reactants_g,
                products_g,
                is_masked,
                masked_labels,
            ) = self.atom_type_masker.mask_features(
                reactants_g,
                products_g,
                reaction,
            )

            # add info to label and meta
            label["masked_atom_type"] = torch.as_tensor(
                masked_labels, dtype=torch.int64
            )
            meta["is_atom_masked"] = torch.as_tensor(is_masked, dtype=torch.bool)

        if self.return_index:
            return item, reactants_g, products_g, reaction_g, meta, label
        else:
            return reactants_g, products_g, reaction_g, meta, label

    @staticmethod
    def collate_fn(samples):
        indices, reactants_g, products_g, reaction_g, metadata, labels = map(
            list, zip(*samples)
        )

        batched_indices = torch.as_tensor(indices)

        batched_molecule_graphs = dgl.batch(reactants_g + products_g)

        if reaction_g[0] is None:
            batched_reaction_graphs = None
        else:
            batched_reaction_graphs = dgl.batch(reaction_g, ndata=None, edata=None)

        # labels
        keys = labels[0].keys()
        batched_labels = {k: torch.cat([d[k] for d in labels]) for k in keys}

        # metadata used to split global and bond features
        keys = metadata[0].keys()
        batched_metadata = {k: [d[k] for d in metadata] for k in keys}

        # convert some metadata to tensor
        for k, v in batched_metadata.items():
            if k in ["atom_hop_dist", "bond_hop_dist", "is_atom_masked"]:
                # each element of v is a 1D tensor
                batched_metadata[k] = torch.cat(v)

        return (
            batched_indices,
            batched_molecule_graphs,
            batched_reaction_graphs,
            batched_labels,
            batched_metadata,
        )


class BaseContrastiveDataset(BaseDataset):
    """
    Base dataset for contrastive learning.
    """

    def __init__(
        self,
        filename: Union[str, Path],
        atom_featurizer: Callable,
        bond_featurizer: Callable,
        global_featurizer: callable,
        *,
        build_reaction_graph: bool = True,
        init_state_dict: Optional[Union[Dict, Path]] = None,
        transform_features: bool = True,
        return_index: bool = True,
        num_processes: int = 1,
        #
        # args to control labels
        #
        transform1: Callable = None,
        transform2: Callable = None,
    ):

        super().__init__(
            filename,
            atom_featurizer,
            bond_featurizer,
            global_featurizer,
            build_reaction_graph=build_reaction_graph,
            init_state_dict=init_state_dict,
            transform_features=transform_features,
            return_index=return_index,
            num_processes=num_processes,
        )

        # labels and metadata (one inner dict for each reaction)
        # do not use [{}] * len(self.reactions); update one will change all
        self.labels = [{} for _ in range(len(self.reactions))]
        self.medadata = [{} for _ in range(len(self.reactions))]

        self.generate_metadata()

        # transforms
        self.transform1 = transform1
        self.transform2 = transform2
        if isinstance(self.transform1, MaskAtomAttribute) or isinstance(
            self.transform2, MaskAtomAttribute
        ):
            self.backup_atom_features()
        if isinstance(self.transform1, MaskBondAttribute) or isinstance(
            self.transform2, MaskBondAttribute
        ):
            self.backup_bond_features()

    def backup_atom_features(self):
        """
        Make a copy of the input features in case the mask features transforms modify
        them.
        """
        self.reactants_atom_features = []
        self.products_atom_features = []
        for reactants_g, products_g, _ in self.dgl_graphs:
            self.reactants_atom_features.append(
                reactants_g.nodes["atom"].data.pop("feat")
            )
            self.products_atom_features.append(
                products_g.nodes["atom"].data.pop("feat")
            )

    def backup_bond_features(self):
        """
        Make a copy of the input features in case the mask features transforms modify
        them.
        """
        self.reactants_bond_features = []
        self.products_bond_features = []
        for reactants_g, products_g, _ in self.dgl_graphs:
            self.reactants_bond_features.append(
                reactants_g.edges["bond"].data.pop("feat")
            )
            self.products_bond_features.append(
                products_g.edges["bond"].data.pop("feat")
            )

    def generate_metadata(self):

        for i, (rxn, label) in enumerate(zip(self.reactions, self.labels)):
            meta = {
                "reactant_num_molecules": len(rxn.reactants),
                "product_num_molecules": len(rxn.products),
            }

            self.medadata[i].update(meta)

    @staticmethod
    def update_meta(meta, reactants_g, products_g, reaction):

        meta["num_atoms"] = reactants_g.num_nodes("atom")

        # Only unchanged bonds are augmented, lost and added bonds are not
        num_lost = len(reaction.lost_bonds)
        num_added = len(reaction.added_bonds)
        num_unchanged = reactants_g.num_edges("bond") // 2 - num_lost
        meta["num_bonds"] = num_unchanged + num_lost + num_added
        meta["num_unchanged_bonds"] = num_unchanged
        meta["num_reactant_bonds"] = num_unchanged + num_lost
        meta["num_product_bonds"] = num_unchanged + num_added

        # bonds in reaction center?
        meta["bonds_in_reaction_center"] = [False] * num_unchanged + [True] * (
            num_lost + num_added
        )

        # atoms in reaction center?
        atom_hop_dist = np.asarray(reaction.atom_distance_to_reaction_center)
        remaining_atoms = reactants_g.nodes["atom"].data["_ID"]
        atom_hop_dist = atom_hop_dist[remaining_atoms]
        in_center = (atom_hop_dist == 0).tolist()
        meta["atoms_in_reaction_center"] = in_center

        return meta

    def __getitem__(self, item: int):
        """
        Get data point with index.
        """
        reactants_g, products_g, reaction_g = self.dgl_graphs[item]
        reaction = self.reactions[item]
        label = self.labels[item]
        meta = self.medadata[item]

        # Assign atom/bond features bach to graph. Should clone to keep backup intact.
        if isinstance(self.transform1, MaskAtomAttribute) or isinstance(
            self.transform2, MaskAtomAttribute
        ):
            reactants_g.nodes["atom"].data["feat"] = self.reactants_atom_features[
                item
            ].clone()
            products_g.nodes["atom"].data["feat"] = self.products_atom_features[
                item
            ].clone()

        if isinstance(self.transform1, MaskBondAttribute) or isinstance(
            self.transform2, MaskBondAttribute
        ):
            reactants_g.edges["bond"].data["feat"] = self.reactants_bond_features[
                item
            ].clone()
            products_g.edges["bond"].data["feat"] = self.products_bond_features[
                item
            ].clone()

        # Augment graph and features
        reactants_g1, products_g1, reaction_g1, _ = self.transform1(
            reactants_g, products_g, reaction_g, reaction
        )
        reactants_g2, products_g2, reaction_g2, _ = self.transform2(
            reactants_g, products_g, reaction_g, reaction
        )

        meta1 = self.update_meta(copy.copy(meta), reactants_g1, products_g1, reaction)
        meta2 = self.update_meta(copy.copy(meta), reactants_g2, products_g2, reaction)

        # #
        # # for profile running speed purpose
        # #
        # reactants_g1, products_g1, reaction_g1 = reactants_g, products_g, reaction_g
        # reactants_g2, products_g2, reaction_g2 = reactants_g, products_g, reaction_g
        # meta1 = self.update_meta(copy.copy(meta), reactants_g1, products_g1)
        # meta2 = self.update_meta(copy.copy(meta), reactants_g2, products_g2)

        return (
            item,
            reactants_g,
            reactants_g1,
            reactants_g2,
            products_g,
            products_g1,
            products_g2,
            reaction_g,
            reaction_g1,
            reaction_g2,
            meta,
            meta1,
            meta2,
            label,
        )

    @staticmethod
    def collate_fn(samples):
        (
            indices,
            reactants_g,
            reactants_g1,
            reactants_g2,
            products_g,
            products_g1,
            products_g2,
            reaction_g,
            reaction_g1,
            reaction_g2,
            metadata,
            metadata1,
            metadata2,
            labels,
        ) = map(list, zip(*samples))

        batched_indices = torch.as_tensor(indices)

        batched_molecule_graphs1 = dgl.batch(reactants_g1 + products_g1)
        batched_molecule_graphs2 = dgl.batch(reactants_g2 + products_g2)

        if reaction_g[0] is None:
            batched_reaction_graphs = None
        else:
            batched_reaction_graphs = dgl.batch(reaction_g, ndata=None, edata=None)

        # labels
        keys = labels[0].keys()
        batched_labels = {k: torch.cat([d[k] for d in labels]) for k in keys}

        # metadata used to split global and bond features
        # keys = metadata[0].keys()
        # batched_metadata = {k: [d[k] for d in metadata] for k in keys}
        keys = metadata1[0].keys()
        batched_metadata1 = {k: [d[k] for d in metadata1] for k in keys}
        batched_metadata2 = {k: [d[k] for d in metadata2] for k in keys}

        return (
            batched_indices,
            (batched_molecule_graphs1, batched_molecule_graphs2),
            batched_reaction_graphs,
            batched_labels,
            (batched_metadata1, batched_metadata2),
        )
