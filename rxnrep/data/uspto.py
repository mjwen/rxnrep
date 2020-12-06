import functools
import logging
import multiprocessing
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import dgl
import numpy as np
import pandas as pd
import torch
from sklearn.utils import class_weight

from rxnrep.core.molecule import Molecule, MoleculeError
from rxnrep.core.reaction import Reaction, ReactionError, smiles_to_reaction
from rxnrep.data.dataset import BaseDataset
from rxnrep.data.grapher import (
    combine_graphs,
    create_hetero_molecule_graph,
    create_reaction_graph,
    get_atom_distance_to_reaction_center,
    get_bond_distance_to_reaction_center,
)
from rxnrep.utils import to_path

logger = logging.getLogger(__name__)


class USPTODataset(BaseDataset):
    """
    USPTO dataset for unsupervised reaction representation.

    Args:
        filename: tsv file of smiles reactions and labels
        atom_featurizer: function to create atom features
        bond_featurizer: function to create bond features
        global_featurizer: function to create global features
        transform_features: whether to standardize the atom, bond, and global features.
            If `True`, each feature column will first subtract the mean and then divide
            by the standard deviation.
        max_hop_distance: maximum allowed hop distance from the reaction center for
            atom and bond. Used to determine atom and bond label
        init_state_dict: initial state dict (or a yaml file of the state dict) containing
            the state of the dataset used for training: including all the atom types in
            the molecules, mean and stdev of the features (if transform_features is
            `True`). If `None`, these properties are computed from the current dataset.
        num_processes: number of processes used to load and process the dataset.
        return_index: whether to return the index of the sample in the dataset
    """

    def __init__(
        self,
        filename: Union[str, Path],
        atom_featurizer: Callable,
        bond_featurizer: Callable,
        global_featurizer: Callable,
        transform_features: bool = True,
        max_hop_distance: int = 3,
        init_state_dict: Optional[Union[Dict, Path]] = None,
        num_processes: int = 1,
        return_index: bool = True,
    ):

        # read input files
        reactions, raw_labels, failed = self.read_file(filename, num_processes)

        super(USPTODataset, self).__init__(
            reactions,
            atom_featurizer,
            bond_featurizer,
            global_featurizer,
            init_state_dict,
            num_processes,
            return_index,
        )

        # set failed and labels
        self._failed = failed
        self._raw_labels = raw_labels

        # convert reactions to dgl graphs
        self.reaction_graphs = self.build_graph_and_featurize()

        if transform_features:
            self.scale_features()

        self.max_hop_distance = max_hop_distance
        self.labels = self.generate_labels()

        self.metadata = {}

    @staticmethod
    def read_file(filename, nprocs):

        # read file
        logger.info("Start reading dataset file...")

        filename = to_path(filename)
        df = pd.read_csv(filename, sep="\t")
        smiles_reactions = df["reaction"].tolist()
        labels = df["label"].to_list()

        logger.info("Finish reading dataset file...")

        # convert to reactions and labels
        logger.info("Start converting to reactions...")

        if nprocs == 1:
            rxn_lb = [
                process_one_reaction_from_input_file(smi, lb, smi + f"_index-{i}")
                for i, (smi, lb) in enumerate(zip(smiles_reactions, labels))
            ]
        else:
            ids = [smi + f"_index-{i}" for i, smi in enumerate(smiles_reactions)]
            args = zip(smiles_reactions, labels, ids)
            with multiprocessing.Pool(nprocs) as p:
                rxn_lb = p.starmap(process_one_reaction_from_input_file, args)
        reactions, labels = map(list, zip(*rxn_lb))

        failed = []
        succeed_reactions = []
        succeed_labels = []
        for rxn, lb in zip(reactions, labels):
            if rxn is None or lb is None:
                failed.append(True)
            else:
                succeed_reactions.append(rxn)
                succeed_labels.append(lb)
                failed.append(False)

        logger.info(
            f"Finish converting to reactions. Number succeed {len(succeed_reactions)}, "
            f"number failed {Counter(failed)[True]}."
        )

        return succeed_reactions, succeed_labels, failed

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
            reaction_graphs = [
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
                reaction_graphs = p.map(func, self.reactions)

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

        return reaction_graphs

    def generate_labels(self) -> List[Dict[str, torch.Tensor]]:
        """
        Labels for all reactions.

        Each dict is the labels for one reaction, with keys:
            `atom_hop_dist` and `bond_hop_dist`.
        """

        labels = []
        for rxn in self.reactions:
            atom_hop = get_atom_distance_to_reaction_center(
                rxn, max_hop=self.max_hop_distance
            )
            bond_hop = get_bond_distance_to_reaction_center(
                rxn, atom_hop_distances=atom_hop, max_hop=self.max_hop_distance
            )
            labels.append(
                {
                    "atom_hop_dist": torch.as_tensor(atom_hop, dtype=torch.int64),
                    "bond_hop_dist": torch.as_tensor(bond_hop, dtype=torch.int64),
                }
            )
        return labels

    def get_class_weight(self) -> Dict[str, torch.Tensor]:
        """
        Create class weight to be used in cross entropy losses.

        This is for labels generated in `generate_labels()`.
        For each type of, it is computed as the mean over all reactions.
        """

        # atom hop class weight

        # Unique labels should be `list(range(atom_hop_num_classes))`,  where
        # `atom_hop_num_classes`could be either 1) `max_hop_distance + 2` or
        # 2) `max_hop_distance + 3` depending on whether there are atoms that are
        # both in lost bonds and added bonds. For 1), there does not exist such atoms,
        # and for 2) there do exist such atoms.
        # The labels are atoms only in lost bond (class 0), atoms in unchanged bond (
        # class 1 to max_hop_distance), added bond (class max_hop_distance + 1),
        # and atoms in both lost and added bonds (class max_hop_distance + 2).
        all_atom_hop_labels = np.concatenate(
            [lb["atom_hop_dist"] for lb in self.labels]
        )

        unique_labels = sorted(set(all_atom_hop_labels))
        if unique_labels != list(
            range(self.max_hop_distance + 2)
        ) and unique_labels != list(range(self.max_hop_distance + 3)):
            raise RuntimeError(
                f"Unable to compute atom class weight; some classes do not have valid "
                f"labels. num_classes: {self.max_hop_distance + 2} unique labels: "
                f"{unique_labels}"
            )

        atom_hop_weight = class_weight.compute_class_weight(
            "balanced",
            classes=unique_labels,
            y=all_atom_hop_labels,
        )

        # bond hop class weight
        # Unique labels should be `list(range(bond_hop_num_classes))`, where
        # `bond_hop_num_classes = max_hop_distance + 2`. Unlike atom hop dist,
        # there are only lost (class 0), unchanged (class 1 to max_hop_distance),
        # and added bonds (class max_hop_distance + 1).
        all_bond_hop_labels = np.concatenate(
            [lb["bond_hop_dist"] for lb in self.labels]
        )

        unique_labels = sorted(set(all_bond_hop_labels))
        if unique_labels != list(range(self.max_hop_distance + 2)):
            raise RuntimeError(
                f"Unable to compute bond class weight; some classes do not have valid "
                f"labels. num_classes: {self.max_hop_distance + 2} unique labels: "
                f"{unique_labels}"
            )

        bond_hop_weight = class_weight.compute_class_weight(
            "balanced",
            classes=unique_labels,
            y=all_bond_hop_labels,
        )

        weight = {
            "atom_hop_dist": torch.as_tensor(atom_hop_weight, dtype=torch.float32),
            "bond_hop_dist": torch.as_tensor(bond_hop_weight, dtype=torch.float32),
        }

        return weight

    def get_molecule_graphs(self) -> List[dgl.DGLGraph]:
        """
        Get all the molecule graphs in the dataset.
        """
        graphs = []
        for reactants_g, products_g, _ in self.reaction_graphs:
            graphs.extend([reactants_g, products_g])

        return graphs

    def __getitem__(self, item: int):
        """
        Get data point with index.
        """
        reactants_g, products_g, reaction_g = self.reaction_graphs[item]
        reaction = self.reactions[item]
        label = self.labels[item]

        # get metadata
        if item in self.metadata:
            meta = self.metadata[item]
        else:
            meta = {
                "reactant_num_molecules": len(reaction.reactants),
                "product_num_molecules": len(reaction.products),
                "num_unchanged_bonds": len(reaction.unchanged_bonds),
                "num_lost_bonds": len(reaction.lost_bonds),
                "num_added_bonds": len(reaction.added_bonds),
            }
            self.metadata[item] = meta

        if self.return_index:
            return item, reactants_g, products_g, reaction_g, meta, label
        else:
            return reactants_g, products_g, reaction_g, meta, label

    def __len__(self) -> int:
        """
        Returns length of dataset (i.e. number of reactions)
        """
        return len(self.reaction_graphs)

    @staticmethod
    def collate_fn(samples):
        indices, reactants_g, products_g, reaction_g, metadata, labels = map(
            list, zip(*samples)
        )

        batched_indices = torch.as_tensor(indices)

        batched_molecule_graphs = dgl.batch(reactants_g + products_g)
        batched_reaction_graphs = dgl.batch(reaction_g, ndata=None, edata=None)

        # labels
        keys = labels[0].keys()
        batched_labels = {k: torch.cat([d[k] for d in labels]) for k in keys}

        # metadata used to split global and bond features
        keys = metadata[0].keys()
        batched_metadata = {k: [d[k] for d in metadata] for k in keys}

        return (
            batched_indices,
            batched_molecule_graphs,
            batched_reaction_graphs,
            batched_labels,
            batched_metadata,
        )


class SchneiderDataset(USPTODataset):
    """
    Schneider 50k USPTO dataset with class labels for reactions.

    The difference between this and the USPTO dataset is that there is class label in
    this dataset and no class label in USPTO. This is added as the `reaction_class`
    in the `labels`.
    """

    def generate_labels(self) -> List[Dict[str, torch.Tensor]]:
        """
        Labels for all reactions.

        Each dict is the labels for one reaction, with keys:
        `atom_hop_dist`, `bond_hop_dist` , and `reaction_class`.
        """

        # labels for atom hop and bond hop
        labels = super(SchneiderDataset, self).generate_labels()

        for rxn_class, rxn_label in zip(self._raw_labels, labels):
            rxn_label["reaction_class"] = torch.as_tensor(
                [int(rxn_class)], dtype=torch.int64
            )
        return labels

    def get_class_weight(
        self, num_reaction_classes: int = 50
    ) -> Dict[str, torch.Tensor]:
        """
        Create class weight to be used in cross entropy losses.

        This is for labels generated in `generate_labels()`.
        For each type of, it is computed as the mean over all reactions.


        The weight of each class is inversely proportional to the number of data points
        in the dataset, i.e.

        n_samples/(n_classes * np.bincount(y))

        Args:
            num_reaction_classes: number of reaction classes in the dataset. The class
            labels should be 0, 1, 2, ... num_reaction_classes-1.
        """
        # class weight for atom hop and bond hop
        weight = super(SchneiderDataset, self).get_class_weight()

        # class weight for reaction classes
        w = class_weight.compute_class_weight(
            "balanced", classes=list(range(num_reaction_classes)), y=self._raw_labels
        )
        w = torch.as_tensor(w, dtype=torch.float32)
        weight["reaction_class"] = w

        return weight


def process_one_reaction_from_input_file(
    smiles_reaction: str, label: str, id: str
) -> Tuple[Union[Reaction, None], Any]:
    # create reaction
    try:
        reaction = smiles_to_reaction(
            smiles_reaction,
            id=id,
            ignore_reagents=True,
            sanity_check=False,
        )
    except (MoleculeError, ReactionError):
        return None, None

    return reaction, label


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
