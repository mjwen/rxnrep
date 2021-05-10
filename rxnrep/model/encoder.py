import logging
from typing import Any, Dict, List, Optional, Tuple

import dgl
import torch
import torch.nn as nn
from dgl.ops import segment_reduce

from rxnrep.model.gatedconv import GatedGCNConv
from rxnrep.model.gin import GINConv, GINConvGlobal, GINConvOriginal
from rxnrep.model.gatconv import GATConv
from rxnrep.model.readout import get_reaction_feature_pooling
from rxnrep.model.utils import MLP, UnifySize

logger = logging.getLogger(__name__)


class ReactionEncoder(nn.Module):
    """
    Representation of a reaction.

    This is achieved by:

    1. update atom, bond, and (optionally global) features of molecule graphs for the
       reactants and products with gated graph conv layers.
    2. compute the difference features between the products and the reactants.
    3. (optional) update the difference features using the reaction graph.
    4. (optional) update difference features using MLP
    5. readout reaction features using a pool, e.g. set2set
    6. (optional) update pooled features using MLP

    Support conv method:
        GatedGCNConv
        GINConvOriginal
        GINConv
        GINConvGlobal
        GATConv

    Args:
        embedding_size: Typically, atom, bond, and global features do not have the same
            size. We apply a linear layer to unity their size to `embedding_size`. If
            `None`, the embedding layer is not applied.
        molecule_conv_layer_sizes: Hidden layer size of the graph conv layers for molecule
            feature update. There will be `len(molecule_hidden_size)` graph conv layers.

        has_global_feats: whether to use global feats in feature update. Note the
            difference between this and pool_global_feats.
        pool_global_feats: whether to add global features to final reaction
            representation. Note the difference between this and has_global_feats. If
            has_global_feats is False, this much be False as well since there is no
            global feats to be used. If has_global_feats is True, this can be either
            True or False.
    """

    def __init__(
        self,
        in_feats: Dict[str, int],
        embedding_size: Optional[int] = None,
        # 1
        molecule_conv_layer_sizes: List[int] = (64, 64),
        molecule_num_fc_layers: int = 2,
        molecule_batch_norm: bool = True,
        molecule_activation: str = "ReLU",
        molecule_residual: bool = True,
        molecule_dropout: float = 0.0,
        # 3
        reaction_conv_layer_sizes: Optional[List[int]] = None,
        reaction_num_fc_layers: Optional[int] = 2,
        reaction_batch_norm: Optional[bool] = True,
        reaction_activation: Optional[str] = "ReLU",
        reaction_residual: Optional[bool] = True,
        reaction_dropout: Optional[float] = 0.0,
        #
        conv="GatedGCNConv",
        has_global_feats: bool = True,
        # how to combine reactants and products graphs: difference, concatenate?
        combine_reactants_products: str = "difference",
        # 4, mlp after combining reactants and products features
        mlp_diff_layer_sizes: Optional[List[int]] = None,
        mlp_diff_layer_batch_norm: Optional[bool] = True,
        mlp_diff_layer_activation: Optional[str] = "ReLU",
        # 5, pool
        pool_method: str = "set2set",
        pool_atom_feats: bool = True,
        pool_bond_feats: bool = True,
        pool_global_feats: bool = True,
        pool_kwargs: Dict[str, Any] = None,
        # 4, mlp after pool
        mlp_pool_layer_sizes: Optional[List[int]] = None,
        mlp_pool_layer_batch_norm: Optional[bool] = True,
        mlp_pool_layer_activation: Optional[str] = "ReLU",
    ):
        super(ReactionEncoder, self).__init__()

        # check input
        if pool_global_feats and not has_global_feats:
            raise ValueError("pool_global_feats=True, while has_global_feats=False")

        self.has_global_feats = has_global_feats

        # when combine reactants and products using concatenate, we do not do it for
        # bonds
        self.combine_reactants_products = combine_reactants_products
        if self.combine_reactants_products == "concatenate":
            assert not reaction_conv_layer_sizes, (
                "Can not do reaction conv when concatenate reactants and products "
                "features"
            )
            assert (
                not pool_bond_feats
            ), "Cannot poll bond feats when concat reactants and products feats"

        # remove global feats (in case the featurizer creates it)
        if not has_global_feats and "global" in in_feats:
            in_feats.pop("global")

        # ========== encoding reaction features ==========

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
            assert has_global_feats, f"Select `{conv}`, but `has_global_feats = False`"
        elif conv == "GINConvGlobal":
            conv_class = GINConvGlobal
            assert has_global_feats, f"Select `{conv}`, but `has_global_feats = False`"
        elif conv == "GINConv":
            conv_class = GINConv
            assert (
                not has_global_feats
            ), f"Select `{conv}`, but `has_global_feats = True`"
        elif conv == "GINConvOriginal":
            conv_class = GINConvOriginal
            assert (
                not has_global_feats
            ), f"Select `{conv}`, but `has_global_feats = True`"
        elif conv == "GATConv":
            conv_class = GATConv
        else:
            raise ValueError(f"Got unexpected conv {conv}")

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
                    out_batch_norm=molecule_batch_norm,
                    out_activation=molecule_activation,
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
                        out_batch_norm=reaction_batch_norm,
                        out_activation=reaction_activation,
                        residual=reaction_residual,
                        dropout=reaction_dropout,
                    )
                )
                in_size = layer_size

            conv_outsize = reaction_conv_layer_sizes[-1]
        else:
            self.reaction_conv_layers = None
            conv_outsize = molecule_conv_layer_sizes[-1]

        # ========== combine reactants and products features ==========
        if self.combine_reactants_products == "difference":
            combine_out_size = conv_outsize
        elif self.combine_reactants_products == "concatenate":
            combine_out_size = 2 * conv_outsize
        else:
            raise ValueError(
                "Not supported method for combine reactants and products "
                f"{self.combine_reactants_products}"
            )

        # ========== mlp diff ==========
        if mlp_diff_layer_sizes:

            self.feat_types = []
            if pool_atom_feats:
                self.feat_types.append("atom")
            if pool_bond_feats:
                self.feat_types.append("bond")
            if pool_global_feats:
                self.feat_types.append("global")

            self.mlp_diff = nn.ModuleDict(
                {
                    k: MLP(
                        in_size=combine_out_size,
                        hidden_sizes=mlp_diff_layer_sizes,
                        batch_norm=mlp_diff_layer_batch_norm,
                        activation=mlp_diff_layer_activation,
                    )
                    for k in self.feat_types
                }
            )

            mlp_diff_outsize = mlp_diff_layer_sizes[-1]
        else:
            self.mlp_diff = None
            mlp_diff_outsize = combine_out_size

        # ========== reaction feature pool ==========
        self.readout = get_reaction_feature_pooling(
            pool_method,
            mlp_diff_outsize,
            pool_atom_feats,
            pool_bond_feats,
            pool_global_feats,
            pool_kwargs,
        )
        pool_out_size = self.readout.out_size

        # ========== mlp after pool ==========
        if mlp_pool_layer_sizes:
            self.mlp_pool = MLP(
                in_size=pool_out_size,
                hidden_sizes=mlp_pool_layer_sizes,
                batch_norm=mlp_pool_layer_batch_norm,
                activation=mlp_pool_layer_activation,
            )

            mlp_pool_outsize = mlp_pool_layer_sizes[-1]
        else:
            self.mlp_pool = None
            mlp_pool_outsize = pool_out_size

        self.node_feats_size = mlp_diff_outsize
        self.reaction_feats_size = mlp_pool_outsize

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

        diff_feats = self.get_difference_feature(
            molecule_graphs, reaction_graphs, feats, metadata
        )

        # reaction graph conv layer
        feats = diff_feats
        if self.reaction_conv_layers:
            for layer in self.reaction_conv_layers:
                feats = layer(reaction_graphs, feats)

        # mlp diff
        if self.mlp_diff:
            feats = {k: self.mlp_diff[k](feats[k]) for k in self.feat_types}

        # readout reaction features, a 1D tensor for each reaction
        reaction_feats = self.readout(molecule_graphs, reaction_graphs, feats, metadata)

        # mlp pool
        if self.mlp_pool:
            reaction_feats = self.mlp_pool(reaction_feats)

        return feats, reaction_feats

    def get_pool_attention_score(
        self,
        molecule_graphs: dgl.DGLGraph,
        reaction_graphs: dgl.DGLGraph,
        feats: Dict[str, torch.Tensor],
        metadata: Dict[str, List[int]],
    ):

        diff_feats = self.get_difference_feature(
            molecule_graphs, reaction_graphs, feats, metadata
        )

        # reaction graph conv layer
        feats = diff_feats
        if self.reaction_conv_layers:
            for layer in self.reaction_conv_layers:
                feats = layer(reaction_graphs, feats)

        # mlp diff
        if self.mlp_diff:
            feats = {k: self.mlp_diff[k](feats[k]) for k in self.feat_types}

        # readout reaction features, a 1D tensor for each reaction
        attn_score = self.readout.get_attention_score(
            molecule_graphs, reaction_graphs, feats, metadata
        )

        return attn_score

    def get_difference_feature(
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
        # remove global feats (in case the featurizer creates it)
        if not self.has_global_feats and "global" in feats:
            feats.pop("global")

        # embedding
        feats = self.embedding(feats)

        # molecule graph conv layer
        for layer in self.molecule_conv_layers:
            feats = layer(molecule_graphs, feats)

        if self.combine_reactants_products == "difference":
            # node as edge graph
            # each bond represented by two edges; select one feat for each bond
            feats["bond"] = feats["bond"][::2]

            # create difference reaction features from molecule features
            rxn_feats = create_diff_reaction_features(
                feats, metadata, self.has_global_feats
            )

            # node as edge graph; make two edge feats for each bond
            rxn_feats["bond"] = torch.repeat_interleave(rxn_feats["bond"], 2, dim=0)

        elif self.combine_reactants_products == "concatenate":
            rxn_feats = create_concat_reaction_features(
                feats, metadata, self.has_global_feats
            )
        else:
            raise ValueError(
                f"Not supported combine reactants products method: "
                f"{self.combine_reactants_products}"
            )

        return rxn_feats


def create_diff_reaction_features(
    molecule_feats: Dict[str, torch.Tensor],
    metadata: Dict[str, List[int]],
    has_global_feats=True,
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
        has_global_feats: whether to compute global features difference

    Returns:
        Difference features between the products and the reactants.
        Will have atom and global features, but not bond features.
        Atom and global feature tensors should have the same shape as the reactant
        feature tensors.
    """
    diff_feats = {
        "atom": get_atom_diff_feats(molecule_feats, metadata),
        "bond": get_bond_diff_feats(molecule_feats, metadata),
    }
    if has_global_feats:
        diff_feats["global"] = get_global_diff_feats(molecule_feats, metadata)

    return diff_feats


def get_atom_diff_feats(molecule_feats, metadata):
    atom_feats = molecule_feats["atom"]

    # Atom difference feats
    size = len(atom_feats) // 2  # same number of atom nodes in reactants and products
    # we can do the below to lines because in the collate fn of dataset, all products
    # graphs are appended to reactants graphs
    reactant_atom_feats = atom_feats[:size]
    product_atom_feats = atom_feats[size:]
    diff_atom_feats = product_atom_feats - reactant_atom_feats

    return diff_atom_feats


def get_bond_diff_feats(molecule_feats, metadata):
    pass

    bond_feats = molecule_feats["bond"]

    num_unchanged_bonds = metadata["num_unchanged_bonds"]
    reactant_num_bonds = metadata["num_reactant_bonds"]
    product_num_bonds = metadata["num_product_bonds"]

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

    return diff_bond_feats


def get_global_diff_feats(molecule_feats, metadata):

    global_feats = molecule_feats["global"]

    reactant_num_molecules = metadata["reactant_num_molecules"]
    product_num_molecules = metadata["product_num_molecules"]

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

    return diff_global_feats


def create_concat_reaction_features(
    molecule_feats: Dict[str, torch.Tensor],
    metadata: Dict[str, List[int]],
    has_global_feats=True,
) -> Dict[str, torch.Tensor]:
    rxn_feats = {
        "atom": get_atom_concat_feats(molecule_feats, metadata),
        # "bond": get_bond_diff_feats(molecule_feats, metadata),
    }
    if has_global_feats:
        rxn_feats["global"] = get_global_concat_feats(molecule_feats, metadata)

    return rxn_feats


def get_atom_concat_feats(molecule_feats, metadata):
    atom_feats = molecule_feats["atom"]

    # Atom difference feats
    size = len(atom_feats) // 2  # same number of atom nodes in reactants and products
    # we can do the below to lines because in the collate fn of dataset, all products
    # graphs are appended to reactants graphs
    reactant_atom_feats = atom_feats[:size]
    product_atom_feats = atom_feats[size:]

    reaction_atom_feats = torch.cat((reactant_atom_feats, product_atom_feats), dim=-1)

    return reaction_atom_feats


#
# def get_bond_diff_feats(molecule_feats, metadata):
#     pass
#
#     bond_feats = molecule_feats["bond"]
#
#     num_unchanged_bonds = metadata["num_unchanged_bonds"]
#     reactant_num_bonds = metadata["num_reactant_bonds"]
#     product_num_bonds = metadata["num_product_bonds"]
#
#     # Bond difference feats
#     total_num_reactant_bonds = sum(reactant_num_bonds)
#     reactant_bond_feats = bond_feats[:total_num_reactant_bonds]
#     product_bond_feats = bond_feats[total_num_reactant_bonds:]
#
#     # feats of each reactant (product), list of 2D tensor
#     reactant_bond_feats = torch.split(reactant_bond_feats, reactant_num_bonds)
#     product_bond_feats = torch.split(product_bond_feats, product_num_bonds)
#
#     # calculate difference feats
#     diff_bond_feats = []
#     for i, (r_ft, p_ft) in enumerate(zip(reactant_bond_feats, product_bond_feats)):
#         n_unchanged = num_unchanged_bonds[i]
#         unchanged_bond_feats = p_ft[:n_unchanged] - r_ft[:n_unchanged]
#         lost_bond_feats = -r_ft[n_unchanged:]
#         added_bond_feats = p_ft[n_unchanged:]
#         feats = torch.cat([unchanged_bond_feats, lost_bond_feats, added_bond_feats])
#         diff_bond_feats.append(feats)
#     diff_bond_feats = torch.cat(diff_bond_feats)
#
#     return diff_bond_feats


def get_global_concat_feats(molecule_feats, metadata):

    global_feats = molecule_feats["global"]

    reactant_num_molecules = metadata["reactant_num_molecules"]
    product_num_molecules = metadata["product_num_molecules"]

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

    reaction_global_feats = torch.cat(
        (mean_reactant_global_feats, mean_product_global_feats), dim=-1
    )

    return reaction_global_feats
