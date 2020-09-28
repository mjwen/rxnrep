import multiprocessing
import functools
import logging
from pathlib import Path
import pandas as pd
import dgl
from rxnrep.core.molecule import Molecule, MoleculeError
from rxnrep.core.reaction import Reaction, smiles_to_reaction
from rxnrep.data.dataset import BaseDataset
from rxnrep.data.grapher import (
    create_hetero_molecule_graph,
    combine_graphs,
    create_reaction_graph,
)
from rxnrep.utils import to_path
from typing import List, Callable, Tuple, Optional, Union, Any

logger = logging.getLogger(__name__)


def process_one_reaction_from_input_file(
    smiles_reaction: str, label: str
) -> Tuple[Union[Reaction, None], Any]:
    # create reaction
    try:
        reaction = smiles_to_reaction(smiles_reaction, smiles_reaction)
    except MoleculeError:
        return None, None

    # TODO process label
    label = label

    return reaction, label


def build_hetero_graph_and_featurize_one_reaction(
    reaction: Reaction,
    atom_featurizer: Callable,
    bond_featurizer: Callable,
    global_featurizer: Callable,
    self_loop=False,
) -> Tuple[dgl.DGLGraph, dgl.DGLGraph, dgl.DGLGraph]:
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
    reaction_g = create_reaction_graph(
        reactants_g,
        products_g,
        reaction.get_num_unchanged_bonds(),
        reaction.get_num_lost_bonds(),
        reaction.get_num_added_bonds(),
        self_loop,
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
            atom_featurizer,
            bond_featurizer,
            global_featurizer,
            state_dict_filename,
            num_processes,
        )

        # set failed and labels
        self._failed = failed
        self.labels = labels

        # convert reactions to dgl graphs
        self.reaction_graphs = self.build_graph_and_featurize()

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

    def __getitem__(self, item: int):
        """Get data point with index.
        """
        reactants_g, products_g, reaction_g = self.reaction_graphs[item]
        reaction = self.reactions[item]
        label = self.labels[item]

        return reactants_g, products_g, reaction_g, reaction, label

    def __len__(self) -> int:
        """
        Returns length of dataset (i.e. number of reactions)
        """
        return len(self.reaction_graphs)


def collate_fn(samples):
    reactants_g, products_g, reaction_g, reactions, labels = map(list, zip(*samples))

    batched_molecule_graphs = dgl.batch(reactants_g + products_g)
    batched_reaction_graphs = dgl.batch(reactants_g)

    # metadata used to split global and bond features
    reactant_num_molecules = []
    product_num_molecules = []
    num_unchanged_bonds = []
    num_lost_bonds = []
    num_added_bonds = []
    for rxn in reactions:
        num_unchanged = rxn.get_num_unchanged_bonds()
        num_lost = rxn.get_num_lost_bonds()
        num_added = rxn.get_num_added_bonds()
        reactant_num_molecules.append(len(rxn.reactants))
        product_num_molecules.append(len(rxn.products))
        num_unchanged_bonds.append(num_unchanged)
        num_lost_bonds.append(num_lost)
        num_added_bonds.append(num_added)
    metadata = {
        "reactant_num_molecules": reactant_num_molecules,
        "product_num_molecules": product_num_molecules,
        "num_unchanged_bonds": num_unchanged_bonds,
        "num_lost_bonds": num_lost_bonds,
        "num_added_bonds": num_added_bonds,
    }

    # TODO batch labels

    return batched_molecule_graphs, batched_reaction_graphs, labels, metadata
