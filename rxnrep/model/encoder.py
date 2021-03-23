import logging
from typing import Any, Dict, List, Optional, Tuple

import dgl
import torch
import torch.nn as nn
from dgl.ops import segment_reduce

from rxnrep.model.gatedconv import GatedGCNConv
from rxnrep.model.gatedconv2 import GatedGCNConv as GatedGCNConv2
from rxnrep.model.readout import Pooling
from rxnrep.model.utils import MLP, UnifySize

logger = logging.getLogger(__name__)


class ReactionEncoder(nn.Module):
    """
    Representation of a reaction.

    This is achieve by:

    1. update atom, bond, and (optionally global) features of molecule graphs for the
       reactants and products with gated graph conv layers.
    2. compute the difference features between the products and the reactants.
    3. (optional) update the difference features using the reaction graph.
    4. (optional) update difference features using MLP
    5. readout reaction features using a pooling, e.g. set2set

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
        molecule_conv_layer_sizes: List[int] = (64, 64),
        molecule_num_fc_layers: int = 2,
        molecule_batch_norm: bool = True,
        molecule_activation: str = "ReLU",
        molecule_residual: bool = True,
        molecule_dropout: float = 0.0,
        reaction_conv_layer_sizes: Optional[List[int]] = None,
        reaction_num_fc_layers: Optional[int] = 2,
        reaction_batch_norm: Optional[bool] = True,
        reaction_activation: Optional[str] = "ReLU",
        reaction_residual: Optional[bool] = True,
        reaction_dropout: Optional[float] = 0.0,
        conv="GatedGCNConv2",
        has_global_feats: bool = True,
        # compressing
        compressing_layer_sizes: Optional[List[int]] = None,
        compressing_layer_batch_norm: Optional[bool] = True,
        compressing_layer_activation: Optional[str] = "ReLU",
        # pooling
        pooling_method: str = "set2set",
        pooling_kwargs: Dict[str, Any] = None,
    ):
        super(ReactionEncoder, self).__init__()
        self.conv = conv

        # set default values
        if isinstance(molecule_activation, str):
            molecule_activation = getattr(nn, molecule_activation)()
        if isinstance(reaction_activation, str):
            reaction_activation = getattr(nn, reaction_activation)()

        # ========== encoding mol features ==========

        #
        # embedding to unify feature size
        #

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
            conv_class = GatedGCNConv
        elif conv == "GatedGCNConv2":
            conv_class = GatedGCNConv2
        else:
            raise ValueError()

        #
        # graph conv layers to update features of molecule graph
        #
        self.molecule_conv_layers = nn.ModuleList()
        for layer_size in molecule_conv_layer_sizes:
            self.molecule_conv_layers.append(
                conv_class(
                    in_size=in_size,
                    out_size=layer_size,
                    num_fc_layers=molecule_num_fc_layers,
                    batch_norm=molecule_batch_norm,
                    activation=molecule_activation,
                    residual=molecule_residual,
                    dropout=molecule_dropout,
                )
            )
            in_size = layer_size

        #
        # graph conv layers to update features of reaction graph
        #
        if reaction_conv_layer_sizes:
            self.reaction_conv_layers = nn.ModuleList()
            for layer_size in reaction_conv_layer_sizes:
                self.reaction_conv_layers.append(
                    conv_class(
                        in_size=in_size,
                        out_size=layer_size,
                        num_fc_layers=reaction_num_fc_layers,
                        batch_norm=reaction_batch_norm,
                        activation=reaction_activation,
                        residual=reaction_residual,
                        dropout=reaction_dropout,
                    )
                )
                in_size = layer_size

            conv_outsize = reaction_conv_layer_sizes[-1]
        else:
            self.reaction_conv_layers = None
            conv_outsize = molecule_conv_layer_sizes[-1]

        # ========== compressor ==========
        if compressing_layer_sizes:
            self.feat_types = ["atom", "bond"]
            if has_global_feats:
                self.feat_types.append("global")

            self.compressor = nn.ModuleDict(
                {
                    k: MLP(
                        in_size=conv_outsize,
                        hidden_sizes=compressing_layer_sizes,
                        batch_norm=compressing_layer_batch_norm,
                        activation=compressing_layer_activation,
                    )
                    for k in self.feat_types
                }
            )

            compressor_outsize = compressing_layer_sizes[-1]
        else:
            self.compressor = None
            compressor_outsize = conv_outsize

        # ========== reaction feature pooling ==========
        self.readout = Pooling(
            compressor_outsize, pooling_method, pooling_kwargs, has_global_feats
        )

        self.node_feats_size = compressor_outsize
        self.reaction_feats_size = self.readout.reaction_feats_size

    def forward(
        self,
        molecule_graphs: dgl.DGLGraph,
        reaction_graphs: dgl.DGLGraph,
        feats: Dict[str, torch.Tensor],
        metadata: Dict[str, List[int]],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
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

        Returns:
            feats: {name, feats} Atom, bond, and global features for the reaction,
                of shape (Na, D'), (Nb', D') and (Ng', D'), respectively. The number of
                the atom features stay the same, while the number of bond and global
                features changes. This happens when we assemble the molecules graphs to
                reaction graphs. Specifically, the number of bond nodes for each reaction
                is the union of the unchanged bonds, lost bonds in reactants, and added
                bonds in products, and the number of global nodes become one for each
                reaction.
            reaction_feats: tensor of shape (B, D'), where b is the batch size,
                and D' is the feature size. One row for each reaction.
        """

        # embedding
        feats = self.embedding(feats)

        # molecule graph conv layer
        for layer in self.molecule_conv_layers:
            feats = layer(molecule_graphs, feats)

        # node as edge graph
        # each bond represented by two edges; select one feat for each bond
        if self.conv == "GatedGCNConv2":
            feats["bond"] = feats["bond"][::2]

        # create difference reaction features from molecule features
        diff_feats = create_reaction_features(feats, metadata)

        # node as edge graph; make two edge feats for each bond
        if self.conv == "GatedGCNConv2":
            diff_feats["bond"] = torch.repeat_interleave(diff_feats["bond"], 2, dim=0)

        # reaction graph conv layer
        feats = diff_feats
        if self.reaction_conv_layers:
            for layer in self.reaction_conv_layers:
                feats = layer(reaction_graphs, feats)

        # compressor
        if self.compressor:
            feats = {k: self.compressor[k](feats[k]) for k in self.feat_types}

        # readout reaction features, a 1D tensor for each reaction
        reaction_feats = self.readout(molecule_graphs, reaction_graphs, feats, metadata)

        return feats, reaction_feats

    def get_diff_feats(
        self,
        molecule_graphs: dgl.DGLGraph,
        reaction_graphs: dgl.DGLGraph,
        feats: Dict[str, torch.Tensor],
        metadata: Dict[str, List[int]],
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
            feats = layer(molecule_graphs, feats)

        # node as edge graph
        # each bond represented by two edges; select one feat for each bond
        if self.conv == "GatedGCNConv2":
            feats["bond"] = feats["bond"][::2]

        # create difference reaction features from molecule features
        diff_feats = create_reaction_features(feats, metadata)

        # node as edge graph; make two edge feats for each bond
        if self.conv == "GatedGCNConv2":
            diff_feats["bond"] = torch.repeat_interleave(diff_feats["bond"], 2, dim=0)

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
    reactant_num_bonds = metadata["num_reactant_bonds"]
    product_num_bonds = metadata["num_product_bonds"]

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

    device = reactant_global_feats.device
    mean_reactant_global_feats = segment_reduce(
        torch.tensor(reactant_num_molecules, device=device),
        reactant_global_feats,
        reducer="sum",
    )
    mean_product_global_feats = segment_reduce(
        torch.tensor(product_num_molecules, device=device),
        product_global_feats,
        reducer="sum",
    )

    diff_global_feats = mean_product_global_feats - mean_reactant_global_feats

    diff_feats = {
        "atom": diff_atom_feats,
        "bond": diff_bond_feats,
        "global": diff_global_feats,
    }

    return diff_feats
