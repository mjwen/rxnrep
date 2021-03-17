from typing import Any, Tuple

import dgl
import numpy as np
import torch

from rxnrep.core.reaction import Reaction


class Transform:
    """
    Base class for transform.
    """

    def __init__(self, ratio: float):
        assert 0 < ratio < 1, f"expect ratio be 0<ratio<1, got {ratio}"
        self.ratio = ratio

    def __call__(
        self, reactants_g, products_g, reaction_g, reaction: Reaction
    ) -> Tuple[dgl.DGLGraph, dgl.DGLGraph, dgl.DGLGraph, Any]:
        """
        Args:
            reactants_g:
            products_g:
            reaction_g:
            reaction:
        Returns:
            augmented_reactants_g:
            agumented_products_g:
            agumented_reaction_g:
            metadata:
        """
        raise NotImplementedError

    @staticmethod
    def _get_atoms_not_in_center(reaction) -> np.ndarray:
        """
        Get atoms not in reaction center.
        """
        (not_in_center,) = np.nonzero(reaction.atom_distance_to_reaction_center)
        return not_in_center

    @staticmethod
    def _get_bonds_not_in_center(reaction) -> np.ndarray:
        """
        Get bonds not in reaction center.
        """
        (not_in_center,) = np.nonzero(reaction.bond_distance_to_reaction_center)
        return not_in_center


class Compose:
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *x):
        for t in self.transforms:
            x = t(*x)
        return x

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class DropAtom(Transform):
    """
    Transformation by dropping atoms. The bonds of the atoms are also dropped.

    Atoms in reaction center are not dropped.

    This assumes the atom is a node.
    """

    def __call__(self, reactants_g, products_g, reaction_g, reaction: Reaction):

        not_in_center = self._get_atoms_not_in_center(reaction)
        n = int(self.ratio * len(not_in_center))

        if n == 0:
            return reactants_g, products_g, reaction_g, None

        else:
            to_drop = np.random.choice(not_in_center, n, replace=False)
            to_keep = sorted(set(range(reaction.num_atoms)) - set(to_drop))

            # extract atom dgl subgraph
            g = reactants_g
            nodes = {k: list(range(g.num_nodes(k))) for k in g.ntypes}
            nodes["atom"] = to_keep
            sub_reactants_g = dgl.node_subgraph(g, nodes)

            g = products_g
            nodes = {k: list(range(g.num_nodes(k))) for k in g.ntypes}
            nodes["atom"] = to_keep
            sub_products_g = dgl.node_subgraph(g, nodes)

            return sub_reactants_g, sub_products_g, reaction_g, None


class DropBond(Transform):
    """
    Transformation by dropping bonds. Atoms are NOT dropped.

    Do not drop bonds in reaction center.

    This assumes the bond is represented by two edges.
    """

    def __call__(self, reactants_g, products_g, reaction_g, reaction: Reaction):

        not_in_center = self._get_bonds_not_in_center(reaction)
        n = int(self.ratio * len(not_in_center))

        if n == 0:
            return reactants_g, products_g, reaction_g, None

        else:
            indices = np.random.choice(not_in_center, n, replace=False)
            x = indices * 2  # each bond has two edges (2i and 2i+1)
            to_drop = set(np.concatenate((x, x + 1)))

            num_reactant_edges = 2 * reaction.num_reactants_bonds
            reactant_bonds_to_keep = sorted(set(range(num_reactant_edges)) - to_drop)

            num_product_edges = 2 * reaction.num_products_bonds
            product_bonds_to_keep = sorted(set(range(num_product_edges)) - to_drop)

            # extract atom dgl subgraph
            g = reactants_g
            edges = {k: list(range(g.num_edges(k))) for k in g.etypes}
            edges["bond"] = reactant_bonds_to_keep
            sub_reactants_g = dgl.edge_subgraph(g, edges, preserve_nodes=True)

            g = products_g
            edges = {k: list(range(g.num_edges(k))) for k in g.etypes}
            edges["bond"] = product_bonds_to_keep
            sub_products_g = dgl.edge_subgraph(g, edges, preserve_nodes=True)

            return sub_reactants_g, sub_products_g, reaction_g, None


class MaskAtomAttribute(Transform):
    """
    Only mask atoms not in the center.

    Args:
        ratio:
        mask_value: The type of values to use for the masked features.
            If `None`, 0 will be used. if `mean+std`, feature mean plus feature std
            is used. if `mean`, feature mean is used.
    """

    def __init__(self, ratio: float, mask_value: str = None):
        super().__init__(ratio)

        self.mask_value = mask_value

    def __call__(
        self,
        reactants_g,
        products_g,
        reaction_g,
        reaction: Reaction,
        feature_mean: torch.Tensor = None,
        feature_std: torch.Tensor = None,
    ):

        not_in_center = self._get_atoms_not_in_center(reaction)
        n = int(self.ratio * len(not_in_center))

        if n == 0:
            return reactants_g, products_g, reaction_g, None

        else:
            selected = sorted(np.random.choice(not_in_center, n, replace=False))

            if self.mask_value is not None:
                assert feature_mean is not None and feature_std is not None, (
                    "Feature mean and std needed to mask atom features when "
                    "`mask_value != None`"
                )

                if self.mask_value == "mean+std":
                    value = feature_mean + feature_std
                elif self.mask_value == "mean":
                    value = feature_mean
                else:
                    raise ValueError(f"Unsupported mask value type: {self.mask_value}")
            else:
                value = 0

            reactants_g.nodes["atom"].data["feat"][selected] = value
            products_g.nodes["atom"].data["feat"][selected] = value

            return reactants_g, products_g, reaction_g, None


class MaskBondAttribute(Transform):
    """
    Only mask bonds not in the center.

    Args:
        ratio:
        mask_value: The type of values to use for the masked features.
            If `None`, 0 will be used. if `mean+std`, feature mean plus feature std
            is used. if `mean`, feature mean is used.
    """

    def __init__(self, ratio: float, mask_value: str = None):
        super().__init__(ratio)

        self.mask_value = mask_value

    def __call__(
        self,
        reactants_g,
        products_g,
        reaction_g,
        reaction: Reaction,
        feature_mean: torch.Tensor = None,
        feature_std: torch.Tensor = None,
    ):
        """

        Args:
            reactants_g:
            products_g:
            reaction_g:
            reaction:
            feature_mean: 1D tensor of bond feature mean
            feature_std: 1D tensor of bond feature std
        """

        not_in_center = self._get_bonds_not_in_center(reaction)
        n = int(self.ratio * len(not_in_center))

        if n == 0:
            return reactants_g, products_g, reaction_g, None

        else:
            indices = np.random.choice(not_in_center, n, replace=False)
            x = indices * 2  # each bond has two edges (2i and 2i+1)
            selected = sorted(np.concatenate((x, x + 1)))

            if self.mask_value is not None:
                assert feature_mean is not None and feature_std is not None, (
                    "Feature mean and std needed to mask atom features when "
                    "`mask_value != None`"
                )

                if self.mask_value == "mean+std":
                    value = feature_mean + feature_std
                elif self.mask_value == "mean":
                    value = feature_mean
                else:
                    raise ValueError(f"Unsupported mask value type: {self.mask_value}")
            else:
                value = 0

            reactants_g.edges["bond"].data["feat"][selected] = value
            products_g.edges["bond"].data["feat"][selected] = value

            return reactants_g, products_g, reaction_g, None
