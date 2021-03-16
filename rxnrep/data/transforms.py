import itertools

import dgl
import numpy as np

from rxnrep.core.reaction import Reaction


class Transform:
    """
    Base class for transform.
    """

    def __init__(self, ratio: float):
        assert 0 < ratio < 1, f"expect ratio be 0<ratio<1, got {ratio}"
        self.ratio = ratio

    def __call__(self, reactants_g, products_g, reaction_g, reaction: Reaction):
        pass


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

        atom_dist = reaction.atom_distance_to_reaction_center
        not_in_center = [atom_dist.index(i) for i in atom_dist if i != 0]
        n = int(self.ratio * len(not_in_center))

        if n == 0:
            return reactants_g, products_g, reaction_g, reaction
        else:
            to_drop = np.random.choice(not_in_center, n, replace=False)
            to_keep = [i for i in range(len(atom_dist)) if i not in to_drop]

            # extract atom dgl subgraph
            g = reactants_g
            nodes = {k: list(range(g.num_nodes(k))) for k in g.ntypes}
            nodes["atom"] = to_keep
            sub_reactants_g = dgl.node_subgraph(g, nodes)

            g = products_g
            nodes = {k: list(range(g.num_nodes(k))) for k in g.ntypes}
            nodes["atom"] = to_keep
            sub_products_g = dgl.node_subgraph(g, nodes)

            return sub_reactants_g, sub_products_g, reactants_g, reaction


class DropBond(Transform):
    """
    Transformation by dropping bonds. Atoms are NOT dropped.

    Do not drop bonds in reaction center.

    This assumes the bond is represented by two edges.
    """

    def __call__(self, reactants_g, products_g, reaction_g, reaction: Reaction):

        # select bonds to drop: not select bonds in reaction center
        bonds_dist = reaction.bond_distance_to_reaction_center
        not_in_center = [bonds_dist.index(i) for i in bonds_dist if i != 0]
        n = int(self.ratio * len(not_in_center))

        if n == 0:
            return reactants_g, products_g, reaction_g, reaction
        else:
            to_drop = np.random.choice(not_in_center, n, replace=False)

            # each bond has two edges
            reactant_bonds_to_keep = list(
                itertools.chain.from_iterable(
                    [
                        [2 * i, 2 * i + 1]
                        for i in range(
                            len(reaction.unchanged_bonds) + len(reaction.lost_bonds)
                        )
                        if i not in to_drop
                    ]
                )
            )
            product_bonds_to_keep = list(
                itertools.chain.from_iterable(
                    [
                        [2 * i, 2 * i + 1]
                        for i in range(
                            len(reaction.unchanged_bonds) + len(reaction.added_bonds)
                        )
                        if i not in to_drop
                    ]
                )
            )

            # extract atom dgl subgraph
            g = reactants_g
            edges = {k: list(range(g.num_edges(k))) for k in g.etypes}
            edges["bond"] = reactant_bonds_to_keep
            sub_reactants_g = dgl.edge_subgraph(g, edges, preserve_nodes=True)

            g = products_g
            edges = {k: list(range(g.num_edges(k))) for k in g.etypes}
            edges["bond"] = product_bonds_to_keep
            sub_products_g = dgl.edge_subgraph(g, edges, preserve_nodes=True)

            return sub_reactants_g, sub_products_g, reactants_g, reaction
