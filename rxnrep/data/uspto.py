import multiprocessing
import functools
import logging
from pathlib import Path
import pandas as pd
import dgl
import torch
from rxnrep.core.molecule import Molecule, MoleculeError
from rxnrep.core.reaction import Reaction, smiles_to_reaction
from rxnrep.data.dataset import BaseDataset
from rxnrep.data.grapher import create_hetero_molecule_graph, combine_graphs
from rxnrep.utils import to_path
from typing import List, Callable, Tuple, Optional, Union, Any

logger = logging.getLogger(__name__)


def process_one_reaction_from_input_file(
    smiles_reaction: str, label: str
) -> Tuple[Reaction, Any]:
    # create reaction
    try:
        reaction = smiles_to_reaction(smiles_reaction, smiles_reaction)
    except MoleculeError:
        return None, None

    # process label
    label = label

    return reaction, label


def build_hetero_graph_and_featurize_one_reaction(
    reaction: Reaction,
    atom_featurizer: Callable,
    bond_featurizer: Callable,
    global_featurizer: Callable,
) -> Tuple[dgl.DGLGraph, dgl.DGLGraph]:
    def featurize_one_mol(m: Molecule):

        rdkit_mol = m.rdkit_mol
        # create graph
        g = create_hetero_molecule_graph(rdkit_mol)

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
    bond_map_number = reaction.get_reactants_bond_map_number()
    reactants = combine_graphs(reactant_graphs, atom_map_number, bond_map_number)

    atom_map_number = reaction.get_products_atom_map_number(zero_based=True)
    bond_map_number = reaction.get_products_atom_map_number()
    products = combine_graphs(product_graphs, atom_map_number, bond_map_number)

    return reactants, products


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
        state_dict_filename: path to a torch pickled file containing the state of the
            dataset used for training, such as all the atom types in the in the molecules,
             mean and stdev of the features (if transform_features if `True`).
        num_processes: number of processes used to load and process the dataset.
    """

    def __init__(
        self,
        filename: Union[str, Path],
        atom_featurizer: Callable,
        bond_featurizer: Callable,
        global_featurizer: Callable,
        transform_features: bool = True,
        state_dict_filename: Optional[Union[str, Path]] = None,
        num_processes: int = 1,
    ):

        # read input files
        reactions, labels, failed = self.read_file(filename, num_processes)

        super(USPTODataset, self).__init__(
            reactions,
            labels,
            atom_featurizer,
            bond_featurizer,
            global_featurizer,
            state_dict_filename,
            num_processes,
        )

        # set failed and labels
        self._failed = failed
        self.labels = labels

        # recovery state info
        if state_dict_filename is not None:
            state_dict_filename = torch.load(str(to_path(state_dict_filename)))
            self.load_state_dict(state_dict_filename)

        self.build_graph_and_featurize()

        if transform_features:
            self.scale_features()

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
        reactions, labels = map(list, *zip(rxn_lb))

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

    def build_graph_and_featurize(self) -> List[Tuple[dgl.DGLGraph, dgl.DGLGraph]]:
        """
        Build DGL graphs for molecules in the reactions and then featurize the molecules.

        Each reaction is represented by two graphs, one for reactants and the other for
        products.

        Returns:
            a list of (reactants, products), reactants and products are dgl graphs.
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
            reactions = [
                build_hetero_graph_and_featurize_one_reaction(
                    rxn,
                    atom_featurizer=atom_featurizer,
                    bond_featurizer=self.bond_featurizer,
                    global_featurizer=self.global_featurizer,
                )
                for rxn in self.raw_reactions
            ]
        else:
            func = functools.partial(
                build_hetero_graph_and_featurize_one_reaction,
                atom_featurizer=atom_featurizer,
                bond_featurizer=self.bond_featurizer,
                global_featurizer=self.global_featurizer,
            )
            with multiprocessing.Pool(self.nprocs) as p:
                reactions = p.map(func, self.reactions)

        self.reactions = reactions

        # keep record of feature size and name info
        self._feature_size = {}
        self._feature_name = {}
        featurizers = {
            "atom": self.atom_featurizer,
            "bond": self.bond_featurizer,
            "global": self.global_featurizer,
        }
        for name, feater in featurizers.items():
            ft_name = feater.feature_name
            ft_size = feater.feature_size
            self._feature_size[name] = ft_size
            self._feature_name[name] = ft_name
            logger.info(f"{name} feature size: {ft_size}")
            logger.info(f"{name} feature name: {ft_name}")

        logger.info("Finish building graphs and featurizing...")

        return self.reactions


def collate_fn(samples):
    # get the reactants, products, and labels of a set of N reactions
    reactants, products, labels = map(list, zip(*samples))

    batched_reactant_graphs = dgl.batch(reactants)
    batched_product_graphs = dgl.batch(products)

    # TODO batch labels

    return batched_reactant_graphs, batched_product_graphs, labels
