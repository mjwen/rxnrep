import multiprocessing
import functools
import logging
import itertools
from pathlib import Path
import pandas as pd
import dgl
import numpy as np
import torch
from rxnrep.core.molecule import Molecule, MoleculeError
from rxnrep.core.reaction import Reaction, smiles_to_reaction
from rxnrep.data.dataset import BaseDataset
from rxnrep.data.grapher import (
    create_hetero_molecule_graph,
    combine_graphs,
    create_reaction_graph,
)
from rxnrep.utils import to_path
from typing import List, Callable, Tuple, Optional, Union, Any, Dict

logger = logging.getLogger(__name__)


def process_one_reaction_from_input_file(
    smiles_reaction: str, label: str
) -> Tuple[Union[Reaction, None], Any]:
    # create reaction
    try:
        reaction = smiles_to_reaction(
            smiles_reaction, smiles_reaction, ignore_reagents=True
        )
    except MoleculeError:
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
    (
        num_unchanged,
        num_lost,
        num_added,
    ) = reaction.get_num_unchanged_lost_and_added_bonds()
    reaction_g = create_reaction_graph(
        reactants_g, products_g, num_unchanged, num_lost, num_added, self_loop,
    )

    return reactants_g, products_g, reaction_g


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
        init_state_dict: Optional[Union[Dict, Path]] = None,
        num_processes: int = 1,
        return_index: bool = True,
    ):

        # read input files
        reactions, labels, failed = self.read_file(filename, num_processes)

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
        self._raw_labels = labels

        # convert reactions to dgl graphs
        self.reaction_graphs = self.build_graph_and_featurize()

        if transform_features:
            self.scale_features()

        self.labels = {}
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
                process_one_reaction_from_input_file(smi, lb)
                for smi, lb in zip(smiles_reactions, labels)
            ]
        else:
            args = zip(smiles_reactions, labels)
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

        logger.info("Finish converting to reactions...")

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

    def get_molecule_graphs(self) -> List[dgl.DGLGraph]:
        """
        Get all the molecule graphs in the dataset.
        """
        graphs = []
        for reactants_g, products_g, _ in self.reaction_graphs:
            graphs.extend([reactants_g, products_g])

        return graphs

    def get_atom_in_reaction_center_and_bond_type_class_weight(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create class weight to be used in cross entropy function for node classification
        problems:

        1. whether atom is in reaction center:
            weight = mean(num_atoms_not_in_center / num_atoms_in_center)
            where mean is taken over all reactions.

            Here, if an atom is in center, it has positive class 1, we adjust its
            weight accordingly. See `self._create_label_atom_in_reaction_center()` for
            more.

        2. bond is unchanged bond, lost bond, or added bond:
            total_num_bonds =
            w_unchanged = (num_lost + num_added) / (2 * total_num_bonds)
            w_lost = (num_unchanged + num_added) / (2 * total_num_bonds)
            w_added = (num_unchanged + num_lost) / (2 * total_num_bonds)

            And then w_unchanged, w_lost, and w_added are averaged over all reactions:
            w_unchanged = mean(w_unchanged)
            w_lost = mean(w_lost)
            w_added = mean(w_added)

            Here, unchanged, lost, and added bonds have class labels 0, 1, and 2
            respectively. See `self._create_label_bond_type()` for more.

        Returns:
            weight_atom_in_reaction_center: a scaler tensor giving the weight for the
                positive class.
            weight_bond_type: a tensor of shape (3,), giving the weight for unchanged
                bonds, lost bonds, and added bonds in sequence.
        """
        # TODO can use multiprocessing

        w_in_center = []
        w_unchanged = []
        w_lost = []
        w_added = []
        for rxn in self.reactions:
            unchanged, lost, added = rxn.get_unchanged_lost_and_added_bonds(
                zero_based=True
            )

            # bond weight
            n_unchanged = len(unchanged)
            n_lost = len(lost)
            n_added = len(added)
            n = n_unchanged + n_lost + n_added
            w_unchanged.append((n_lost + n_added) / (2 * n))
            w_lost.append((n_unchanged + n_added) / (2 * n))
            w_added.append((n_unchanged + n_lost) / (2 * n))

            # atom weight
            num_atoms = sum([m.num_atoms for m in rxn.reactants])
            changed_atoms = set(
                [i for i in itertools.chain.from_iterable(lost + added)]
            )
            num_changed = len(changed_atoms)
            if num_changed == 0:
                w_in_center.append(1.0)
            else:
                w_in_center.append((num_atoms - num_changed) / num_changed)

        weight_atom_in_reaction_center = torch.as_tensor(
            np.mean(w_in_center), dtype=torch.float32
        )

        weight_bond_type = [np.mean(w_unchanged), np.mean(w_lost), np.mean(w_added)]
        weight_bond_type = torch.as_tensor(
            np.asarray(weight_bond_type), dtype=torch.float32
        )

        return weight_atom_in_reaction_center, weight_bond_type

    @staticmethod
    def _create_label_bond_type(reaction) -> torch.Tensor:
        """
        Label for bond type classification:
        0: unchanged bond, 1: lost bond, 2: added bond

        Args:
            reaction: th3 reaction

        Returns:
            1D tensor of the class for each bond. The order is the same as the bond
            nodes in the reaction graph.
       """
        result = reaction.get_num_unchanged_lost_and_added_bonds()
        num_unchanged, num_lost, num_added = result

        # Note, reaction graph bond nodes are ordered in the sequence of unchanged bonds,
        # lost bonds, and added bonds in `create_reaction_graph()`
        bond_type = [0] * num_unchanged + [1] * num_lost + [2] * num_added
        bond_type = torch.as_tensor(bond_type, dtype=torch.int64)

        return bond_type

    @staticmethod
    def _create_label_atom_in_reaction_center(reaction: Reaction) -> torch.Tensor:
        """
        Label for atom in reaction center classification:

        Atoms associated with lost bonds in reactants or added bonds in products are in
        the reaction center, given a class label 1.
        Other atoms given a class label 0.

        Args:
            reaction: the reaction

        Returns:
            1D tensor of the class for each atom. The order is the same as the atom
            nodes in the reaction graph.
        """
        num_atoms = sum([m.num_atoms for m in reaction.reactants])

        _, lost, added = reaction.get_unchanged_lost_and_added_bonds(zero_based=True)
        changed_atoms = set([i for i in itertools.chain.from_iterable(lost + added)])

        in_reaction_center = torch.zeros(num_atoms)
        for i in changed_atoms:
            in_reaction_center[i] = 1

        return in_reaction_center

    def __getitem__(self, item: int):
        """Get data point with index.
        """
        reactants_g, products_g, reaction_g = self.reaction_graphs[item]
        reaction = self.reactions[item]

        # get labels, create it if it does not exist
        if item in self.labels:
            labels = self.labels[item]
        else:
            labels = {
                "bond_type": self._create_label_bond_type(reaction),
                "atom_in_reaction_center": self._create_label_atom_in_reaction_center(
                    reaction
                ),
            }
            self.labels[item] = labels

        # get metadata
        if item in self.metadata:
            meta = self.metadata[item]
        else:
            (
                num_unchanged,
                num_lost,
                num_added,
            ) = reaction.get_num_unchanged_lost_and_added_bonds()
            meta = {
                "reactant_num_molecules": len(reaction.reactants),
                "product_num_molecules": len(reaction.products),
                "num_unchanged_bonds": num_unchanged,
                "num_lost_bonds": num_lost,
                "num_added_bonds": num_added,
            }
            self.metadata[item] = meta

        if self.return_index:
            return item, reactants_g, products_g, reaction_g, meta, labels
        else:
            return reactants_g, products_g, reaction_g, meta, labels

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

    def __getitem__(self, item: int):
        """
        Get data point with index.
        """
        reactants_g, products_g, reaction_g = self.reaction_graphs[item]
        reaction = self.reactions[item]

        # get labels, create it if it does not exist
        if item in self.labels:
            labels = self.labels[item]
        else:
            labels = {
                "bond_type": self._create_label_bond_type(reaction),
                "atom_in_reaction_center": self._create_label_atom_in_reaction_center(
                    reaction
                ),
                "reaction_class": torch.as_tensor(int(self._raw_labels[item])),
            }
            self.labels[item] = labels

        # get metadata
        if item in self.metadata:
            meta = self.metadata[item]
        else:
            (
                num_unchanged,
                num_lost,
                num_added,
            ) = reaction.get_num_unchanged_lost_and_added_bonds()
            meta = {
                "reactant_num_molecules": len(reaction.reactants),
                "product_num_molecules": len(reaction.products),
                "num_unchanged_bonds": num_unchanged,
                "num_lost_bonds": num_lost,
                "num_added_bonds": num_added,
            }
            self.metadata[item] = meta

        if self.return_index:
            return item, reactants_g, products_g, reaction_g, meta, labels
        else:
            return reactants_g, products_g, reaction_g, meta, labels
