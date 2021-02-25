#
# A comprehensive model contains all decoders.
#
# encoder:
# - mol conv layers
# - diff mol feats
# - rxn conv layers (could be 0)
# - compression layers (could be 0)
# - pooling (set2set, hot distance)
#
# decoders:
# - atom hop dist
# - bond hop dist
# - masked atom hop
# - reaction clustering
#
# - reaction energy
# - activation energy
# - bep activation energy label

# By default, the compressing layers and all decoders are set to None, meaning they
# will not be used. Also set the hidden_layer_sizes set an empty list will also not use
# the decoder.
#


from typing import Any, Dict, Tuple

import dgl
import torch
import torch.nn as nn

from rxnrep.model.decoder import (
    ActivationEnergyDecoder,
    AtomHopDistDecoder,
    AtomTypeDecoder,
    BondHopDistDecoder,
    ReactionEnergyDecoder,
)
from rxnrep.model.encoder import ReactionEncoder
from rxnrep.model.readout import CompressingNN, HopDistancePooling, Set2SetThenCat


class EncoderAndPooling(nn.Module):
    """
    Encoder and reaction feature pooling part of the model. Add decoder to use this.
    """

    def __init__(
        self,
        in_feats,
        embedding_size,
        *,
        # encoder
        molecule_conv_layer_sizes,
        molecule_num_fc_layers,
        molecule_batch_norm,
        molecule_activation,
        molecule_residual,
        molecule_dropout,
        reaction_conv_layer_sizes,
        reaction_num_fc_layers,
        reaction_batch_norm,
        reaction_activation,
        reaction_residual,
        reaction_dropout,
        # compressing
        compressing_layer_sizes=None,
        compressing_layer_activation=None,
        # pooling
        pooling_method="set2set",
        pooling_kwargs: Dict[str, Any] = None,
    ):

        super().__init__()

        # ========== encoder ==========
        self.encoder = ReactionEncoder(
            in_feats=in_feats,
            embedding_size=embedding_size,
            molecule_conv_layer_sizes=molecule_conv_layer_sizes,
            molecule_num_fc_layers=molecule_num_fc_layers,
            molecule_batch_norm=molecule_batch_norm,
            molecule_activation=molecule_activation,
            molecule_residual=molecule_residual,
            molecule_dropout=molecule_dropout,
            reaction_conv_layer_sizes=reaction_conv_layer_sizes,
            reaction_num_fc_layers=reaction_num_fc_layers,
            reaction_batch_norm=reaction_batch_norm,
            reaction_activation=reaction_activation,
            reaction_residual=reaction_residual,
            reaction_dropout=reaction_dropout,
        )

        # have reaction conv layer
        if reaction_conv_layer_sizes:
            encoder_outsize = reaction_conv_layer_sizes[-1]
        # does not have reaction conv layer
        else:
            encoder_outsize = molecule_conv_layer_sizes[-1]

        # ========== compressor ==========
        if compressing_layer_sizes:
            self.compressor = nn.ModuleDict(
                {
                    k: CompressingNN(
                        in_size=encoder_outsize,
                        hidden_sizes=compressing_layer_sizes,
                        activation=compressing_layer_activation,
                    )
                    for k in ["atom", "bond", "global"]
                }
            )
            compressor_outsize = compressing_layer_sizes[-1]
        else:
            self.compressor = nn.ModuleDict(
                {k: nn.Identity() for k in ["atom", "bond", "global"]}
            )
            compressor_outsize = encoder_outsize

        # ========== reaction feature pooling ==========
        # readout reaction features, one 1D tensor for each reaction

        self.pooling_method = pooling_method

        if pooling_method == "set2set":
            if pooling_kwargs is None:
                set2set_num_iterations = 6
                set2set_num_layers = 3
            else:
                set2set_num_iterations = pooling_kwargs["set2set_num_iterations"]
                set2set_num_layers = pooling_kwargs["set2set_num_layers"]

            in_sizes = [compressor_outsize] * 2
            self.set2set = Set2SetThenCat(
                num_iters=set2set_num_iterations,
                num_layers=set2set_num_layers,
                ntypes=["atom", "bond"],
                in_feats=in_sizes,
                ntypes_direct_cat=["global"],
            )

            pooling_outsize = compressor_outsize * 5

        elif pooling_method == "hop_distance":
            if pooling_kwargs is None:
                raise RuntimeError(
                    "`max_hop_distance` should be provided as `pooling_kwargs` to use "
                    "`hop_distance_pool`"
                )
            else:
                max_hop_distance = pooling_kwargs["max_hop_distance"]
                self.hop_dist_pool = HopDistancePooling(max_hop=max_hop_distance)

            pooling_outsize = compressor_outsize * 3

        elif pooling_method == "global_only":
            pooling_outsize = compressor_outsize

        else:
            raise ValueError(f"Unsupported pooling method `{pooling_method}`")

        self.node_feats_size = compressor_outsize
        self.reaction_feats_size = pooling_outsize

    def forward(
        self,
        molecule_graphs: dgl.DGLGraph,
        reaction_graphs: dgl.DGLGraph,
        feats: Dict[str, torch.Tensor],
        metadata: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        We let forward only returns features and the part to map the features to logits
        in another function: `decode`.

        Args:
            molecule_graphs:
            reaction_graphs:
            feats:
            metadata:
        """
        # encoder
        feats = self.encoder(molecule_graphs, reaction_graphs, feats, metadata)

        # compressor
        feats = {k: self.compressor[k](feats[k]) for k in ["atom", "bond", "global"]}

        # readout reaction features, a 1D tensor for each reaction
        if self.pooling_method == "set2set":
            reaction_feats = self.set2set(reaction_graphs, feats)

        elif self.pooling_method == "hop_distance":

            hop_dist = {
                "atom": metadata["atom_hop_dist"],
                "bond": metadata["bond_hop_dist"],
            }
            reaction_feats = self.hop_dist_pool(reaction_graphs, feats, hop_dist)

        elif self.pooling_method == "global_only":
            reaction_feats = feats["global"]

        else:
            raise ValueError(f"Unsupported pooling method `{self.pooling_method}`")

        return feats, reaction_feats

    def get_diff_feats(
        self,
        molecule_graphs: dgl.DGLGraph,
        reaction_graphs: dgl.DGLGraph,
        feats: Dict[str, torch.Tensor],
        metadata: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Get the atom/bond/global difference features before applying reaction
        convolution.

        Returns:
            {atom:feats, bond:feats, global:feats}

        """

        return self.encoder.get_diff_feats(
            molecule_graphs, reaction_graphs, feats, metadata
        )


class ReactionRepresentation(EncoderAndPooling):
    """
    Model to represent chemical reactions.
    """

    def __init__(
        self,
        in_feats,
        embedding_size,
        *,
        # encoder
        molecule_conv_layer_sizes,
        molecule_num_fc_layers,
        molecule_batch_norm,
        molecule_activation,
        molecule_residual,
        molecule_dropout,
        reaction_conv_layer_sizes,
        reaction_num_fc_layers,
        reaction_batch_norm,
        reaction_activation,
        reaction_residual,
        reaction_dropout,
        # compressing
        compressing_layer_sizes=None,
        compressing_layer_activation=None,
        # pooling
        pooling_method="set2set",
        pooling_kwargs: Dict[str, Any] = None,
        # bond hop dist decoder
        bond_hop_dist_decoder_hidden_layer_sizes=None,
        bond_hop_dist_decoder_activation=None,
        bond_hop_dist_decoder_num_classes=None,
        # atom hop dist decoder
        atom_hop_dist_decoder_hidden_layer_sizes=None,
        atom_hop_dist_decoder_activation=None,
        atom_hop_dist_decoder_num_classes=None,
        # masked atom type decoder
        masked_atom_type_decoder_hidden_layer_sizes=None,
        masked_atom_type_decoder_activation=None,
        masked_atom_type_decoder_num_classes=None,
        # reaction energy decoder
        reaction_energy_decoder_hidden_layer_sizes=None,
        reaction_energy_decoder_activation=None,
        # activation energy decoder
        activation_energy_decoder_hidden_layer_sizes=None,
        activation_energy_decoder_activation=None,
    ):

        # encoder and pooling
        super().__init__(
            in_feats,
            embedding_size,
            molecule_conv_layer_sizes=molecule_conv_layer_sizes,
            molecule_num_fc_layers=molecule_num_fc_layers,
            molecule_batch_norm=molecule_batch_norm,
            molecule_activation=molecule_activation,
            molecule_residual=molecule_residual,
            molecule_dropout=molecule_dropout,
            reaction_conv_layer_sizes=reaction_conv_layer_sizes,
            reaction_num_fc_layers=reaction_num_fc_layers,
            reaction_batch_norm=reaction_batch_norm,
            reaction_activation=reaction_activation,
            reaction_residual=reaction_residual,
            reaction_dropout=reaction_dropout,
            compressing_layer_sizes=compressing_layer_sizes,
            compressing_layer_activation=compressing_layer_activation,
            pooling_method=pooling_method,
            pooling_kwargs=pooling_kwargs,
        )

        # ========== node level decoder ==========

        # bond hop dist decoder
        if bond_hop_dist_decoder_hidden_layer_sizes:
            self.bond_hop_dist_decoder = BondHopDistDecoder(
                in_size=self.node_feats_size,
                num_classes=bond_hop_dist_decoder_num_classes,
                hidden_layer_sizes=bond_hop_dist_decoder_hidden_layer_sizes,
                activation=bond_hop_dist_decoder_activation,
            )
        else:
            self.bond_hop_dist_decoder = None

        # atom hop dist decoder
        if atom_hop_dist_decoder_hidden_layer_sizes:
            self.atom_hop_dist_decoder = AtomHopDistDecoder(
                in_size=self.node_feats_size,
                num_classes=atom_hop_dist_decoder_num_classes,
                hidden_layer_sizes=atom_hop_dist_decoder_hidden_layer_sizes,
                activation=atom_hop_dist_decoder_activation,
            )
        else:
            self.atom_hop_dist_decoder = None

        # masked atom type decoder
        if masked_atom_type_decoder_hidden_layer_sizes:
            self.masked_atom_type_decoder = AtomTypeDecoder(
                in_size=self.node_feats_size,
                num_classes=masked_atom_type_decoder_num_classes,
                hidden_layer_sizes=masked_atom_type_decoder_hidden_layer_sizes,
                activation=masked_atom_type_decoder_activation,
            )
        else:
            self.masked_atom_type_decoder = None

        # ========== reaction level decoder ==========

        # reaction energy decoder
        if reaction_energy_decoder_hidden_layer_sizes:
            self.reaction_energy_decoder = ReactionEnergyDecoder(
                in_size=self.reaction_feats_size,
                hidden_layer_sizes=reaction_energy_decoder_hidden_layer_sizes,
                activation=reaction_energy_decoder_activation,
            )
        else:
            self.reaction_energy_decoder = None

        # activation energy decoder
        if activation_energy_decoder_hidden_layer_sizes:
            self.activation_energy_decoder = ActivationEnergyDecoder(
                in_size=self.reaction_feats_size,
                hidden_layer_sizes=activation_energy_decoder_hidden_layer_sizes,
                activation=activation_energy_decoder_activation,
            )
        else:
            self.activation_energy_decoder = None

    def decode(
        self,
        feats: torch.Tensor,
        reaction_feats: torch.Tensor,
        metadata: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Decode the molecule and reaction features to properties.

        Args:
            feats: first output of `forward()`
            reaction_feats: second output of `forward()`
            metadata:

        Returns:
            predictions: {decoder_name: value} predictions of the decoders.
        """

        # bond hop dist decoder
        bond_hop_dist = (
            None
            if self.bond_hop_dist_decoder is None
            else self.bond_hop_dist_decoder(feats["bond"])
        )

        # atom hop dist decoder
        atom_hop_dist = (
            None
            if self.atom_hop_dist_decoder is None
            else self.atom_hop_dist_decoder(feats["atom"])
        )

        # masked atom type decoder
        if self.masked_atom_type_decoder is None:
            masked_atom_type = None
        else:
            atom_ft = feats["atom"]
            masked_or_not = metadata["is_atom_masked"]
            atom_ft_of_masked_atoms = atom_ft[masked_or_not]
            masked_atom_type = self.masked_atom_type_decoder(atom_ft_of_masked_atoms)

        # reaction decoder
        reaction_energy = (
            None
            if self.reaction_energy_decoder is None
            else self.reaction_energy_decoder(reaction_feats)
        )
        activation_energy = (
            None
            if self.activation_energy_decoder is None
            else self.activation_energy_decoder(reaction_feats)
        )

        # predictions
        predictions = {
            "bond_hop_dist": bond_hop_dist,
            "atom_hop_dist": atom_hop_dist,
            "masked_atom_type": masked_atom_type,
            "reaction_cluster": reaction_feats,
            "reaction_energy": reaction_energy,
            "activation_energy": activation_energy,
        }

        return predictions
