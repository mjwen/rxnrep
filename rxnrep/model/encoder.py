import logging
from typing import Dict, List, Optional

import dgl
import torch
import torch.nn as nn

from rxnrep.model.gatedconv import GatedGCNConv
from rxnrep.model.utils import UnifySize

logger = logging.getLogger(__name__)


class ReactionEncoder(nn.Module):
    """
    Representation of a reaction.

    This is achieve by:

    1. update atom, bond, and global features of molecule graphs for the reactants and
       products with gated graph conv layers.
    2. compute the difference features between the products and the reactants.
    3. update the difference features obtained in step 2 of the reaction graph with gated
       graph conv layers.

    Args:

        embedding_size: Typically, atom, bond, and global features do not have the same
            size. We apply a linear layer to unity their size to `embedding_size`. If
            `None`, the embedding layer is not applied.
        molecule_conv_layer_sizes: Hidden layer size of the graph conv layers for molecule
            feature update. There will be `len(molecule_hidden_size)` graph conv layers.
    """

    def __init__(
        self,
        in_feats: Dict[str, int],
        embedding_size: Optional[int] = None,
        molecule_conv_layer_sizes: Optional[List[int]] = (64, 64),
        molecule_num_fc_layers: Optional[int] = 2,
        molecule_graph_norm: Optional[bool] = False,
        molecule_batch_norm: Optional[bool] = True,
        molecule_activation: Optional[str] = "ReLU",
        molecule_residual: Optional[bool] = True,
        molecule_dropout: Optional[float] = 0.0,
        reaction_conv_layer_sizes: Optional[List[int]] = (64, 64),
        reaction_num_fc_layers: Optional[int] = 2,
        reaction_graph_norm: Optional[bool] = False,
        reaction_batch_norm: Optional[bool] = True,
        reaction_activation: Optional[str] = "ReLU",
        reaction_residual: Optional[bool] = True,
        reaction_dropout: Optional[float] = 0.0,
        conv="GatedGCNConv",
    ):
        super(ReactionEncoder, self).__init__()

        # set default values
        if isinstance(molecule_activation, str):
            molecule_activation = getattr(nn, molecule_activation)()
        if isinstance(reaction_activation, str):
            reaction_activation = getattr(nn, reaction_activation)()

        # unify atom, bond, and global feature size
        if embedding_size is not None:
            self.embedding = UnifySize(in_feats, embedding_size)
            in_size = embedding_size
        else:
            if len(set(in_feats.values())) != 1:
                raise ValueError(
                    f"Expect `in_feats` values be equal to each; got {in_feats}. "
                    "You probably need to set `embedding_size`."
                )
            self.embedding = nn.Identity()
            in_size = list(in_feats.keys())[0]

        # graph conv layer type
        if conv == "GatedGCNConv":
            conv_fn = GatedGCNConv
        else:
            raise ValueError()

        # graph conv layers to update features of molecule graph
        self.molecule_conv_layers = nn.ModuleList()
        for layer_size in molecule_conv_layer_sizes:
            self.molecule_conv_layers.append(
                conv_fn(
                    input_dim=in_size,
                    output_dim=layer_size,
                    num_fc_layers=molecule_num_fc_layers,
                    graph_norm=molecule_graph_norm,
                    batch_norm=molecule_batch_norm,
                    activation=molecule_activation,
                    residual=molecule_residual,
                    dropout=molecule_dropout,
                )
            )
            in_size = layer_size

        # graph conv layers to update features of reaction graph
        self.reaction_conv_layers = nn.ModuleList()
        for layer_size in reaction_conv_layer_sizes:
            self.reaction_conv_layers.append(
                conv_fn(
                    input_dim=in_size,
                    output_dim=layer_size,
                    num_fc_layers=reaction_num_fc_layers,
                    graph_norm=reaction_graph_norm,
                    batch_norm=reaction_batch_norm,
                    activation=reaction_activation,
                    residual=reaction_residual,
                    dropout=reaction_dropout,
                )
            )
            in_size = layer_size

    def forward(
        self,
        molecule_graphs: dgl.DGLGraph,
        reaction_graphs: dgl.DGLGraph,
        feats: Dict[str, torch.Tensor],
        metadata: Dict[str, List[int]],
        norm_atom=None,
        norm_bond=None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            molecule_graphs: (batched) dgl graphs for reactants and products
            reaction_graphs: (batched) dgl graphs for reactions
            feats: (batched) `atom`, `bond`, and `global` features for the molecules.
                The atom features have a shape
                of (Na, D), the bond features have a shape of (Nb, D), and the global
                features have a shape of (Ng, D), where Na is the total number of atom
                nodes in all the reactants and products graphs, Nb is the total number
                of bond nodes in all the reactants and products graphs, and Ng is the
                total number of global nodes. Note that for one reaction. The number of
                nodes for each node type for each reaction is given in the metadata.
            metadata: holds the number of nodes for all reactions. The size of each
                list is equal to the number of reactions, and element `i` gives the info
                for reaction `i`.
            norm_atom: (2D tensor or None) graph norm for atom
            norm_bond: (2D tensor or None) graph norm for bond

        Returns:
            Atom, bond, and global features for the reaction, of shape (Na, D'), (Nb', D')
            and (Ng', D'), respectively. The number of the atom features stay the same,
            while the number of bond and global features changes. This happens when we
            assemble the molecules graphs to reaction graphs. Specifically, the number
            of bond nodes for each reaction is the union of the unchanged bonds,
            lost bonds in reactants, and added bonds in products, and the number of
            global nodes become 1 for each reaction.
        """

        # embedding
        feats = self.embedding(feats)

        # molecule graph conv layer
        for layer in self.molecule_conv_layers:
            feats = layer(molecule_graphs, feats, norm_atom, norm_bond)

        # create difference reaction features from molecule features
        diff_feats = create_reaction_features(feats, metadata)

        # reaction graph conv layer
        feats = diff_feats
        for layer in self.reaction_conv_layers:
            feats = layer(reaction_graphs, feats, norm_atom, norm_bond)

        return feats

    def get_diff_feats(
        self,
        molecule_graphs: dgl.DGLGraph,
        reaction_graphs: dgl.DGLGraph,
        feats: Dict[str, torch.Tensor],
        metadata: Dict[str, List[int]],
        norm_atom=None,
        norm_bond=None,
    ) -> Dict[str, torch.Tensor]:
        """
        Get the difference features before applying reaction conv layers.

        With this, we can check the idea that atoms/bonds far away from the reaction
        center have small (in magnitude) difference features.

        This is the same as the forward function, without the reaction conv layers.
        """

        # embedding
        feats = self.embedding(feats)

        # molecule graph conv layer
        for layer in self.molecule_conv_layers:
            feats = layer(molecule_graphs, feats, norm_atom, norm_bond)

        # create difference reaction features from molecule features
        diff_feats = create_reaction_features(feats, metadata)

        return diff_feats


def create_reaction_features(
    molecule_feats: Dict[str, torch.Tensor], metadata: Dict[str, List[int]]
) -> Dict[str, torch.Tensor]:
    """
    Compute the difference features between the products and the reactants.

    Atom features of the reactants are subtracted from that of the products.

    Bonds features have three parts. For unchanged bonds, reactants features are
    subtracted from the products. For lost bonds in the reactants, the negative of the
    features are used. For added bonds in the products, the features are directly copied.

    Each reaction has 1 global reactions: computed as the difference of the sum of the
    global features between the products and the reactants.

    Args:
        molecule_feats: (batched) features for the molecules.
        metadata: holds the number of nodes for all reactions. The size of each
            list is equal to the number of reactions, and element `i` gives the info
            for reaction `i`.

    Returns:
        Difference features between the products and the reactants.
        Will have atom and global features, but not bond features.
        Atom and global feature tensors should have the same shape as the reactant
        feature tensors.
    """

    atom_feats = molecule_feats["atom"]
    bond_feats = molecule_feats["bond"]
    global_feats = molecule_feats["global"]

    reactant_num_molecules = metadata["reactant_num_molecules"]
    product_num_molecules = metadata["product_num_molecules"]
    num_unchanged_bonds = metadata["num_unchanged_bonds"]
    num_lost_bonds = metadata["num_lost_bonds"]
    num_added_bonds = metadata["num_added_bonds"]
    reactant_num_bonds = [i + j for i, j in zip(num_unchanged_bonds, num_lost_bonds)]
    product_num_bonds = [i + j for i, j in zip(num_unchanged_bonds, num_added_bonds)]

    # Atom difference feats
    size = len(atom_feats) // 2  # same number of atom nodes in reactants and products
    # we can do the below to lines because in the collate fn of dataset, all products
    # graphs are appended to reactants graphs
    reactant_atom_feats = atom_feats[:size]
    product_atom_feats = atom_feats[size:]
    diff_atom_feats = product_atom_feats - reactant_atom_feats

    # Bond difference feats
    total_num_reactant_bonds = sum(reactant_num_bonds)
    reactant_bond_feats = bond_feats[:total_num_reactant_bonds]
    product_bond_feats = bond_feats[total_num_reactant_bonds:]

    # feats of each reactant (product), list of 2D tensor
    reactant_bond_feats = torch.split(reactant_bond_feats, reactant_num_bonds)
    product_bond_feats = torch.split(product_bond_feats, product_num_bonds)

    # calculate difference feats
    diff_bond_feats = []
    for i, (r_ft, p_ft) in enumerate(zip(reactant_bond_feats, product_bond_feats)):
        n_unchanged = num_unchanged_bonds[i]
        unchanged_bond_feats = p_ft[:n_unchanged] - r_ft[:n_unchanged]
        lost_bond_feats = -r_ft[n_unchanged:]
        added_bond_feats = p_ft[n_unchanged:]
        feats = torch.cat([unchanged_bond_feats, lost_bond_feats, added_bond_feats])
        diff_bond_feats.append(feats)
    diff_bond_feats = torch.cat(diff_bond_feats)

    # Global difference feats

    total_num_reactant_molecules = sum(reactant_num_molecules)
    reactant_global_feats = global_feats[:total_num_reactant_molecules]
    product_global_feats = global_feats[total_num_reactant_molecules:]

    # the mean of global features in each reactant (product) graph
    # Note, each reactant (product) graph holds all the molecules in the
    # reactant (products), and thus has multiple global features.

    # The below commented one does not work on GPU since it creates a graph on CPU. If
    # we want to make it work, we can modify dgl.ops.segment.segment_reduce
    # TODO benchmark segment_reduce and torch.split one. report bug to dgl

    # mean_reactant_global_feats = dgl.ops.segment.segment_reduce(
    # torch.tensor(reactant_num_molecules), reactant_global_feats, reducer="sum",
    # )
    # mean_product_global_feats = dgl.ops.segment.segment_reduce(
    # torch.tensor(product_num_molecules), product_global_feats, reducer="sum",
    # )

    split = torch.split(reactant_global_feats, reactant_num_molecules)
    mean_reactant_global_feats = torch.stack([torch.sum(x, dim=0) for x in split])
    split = torch.split(product_global_feats, product_num_molecules)
    mean_product_global_feats = torch.stack([torch.sum(x, dim=0) for x in split])
    diff_global_feats = mean_product_global_feats - mean_reactant_global_feats

    diff_feats = {
        "atom": diff_atom_feats,
        "bond": diff_bond_feats,
        "global": diff_global_feats,
    }

    return diff_feats
