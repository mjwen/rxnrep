import dgl
import numpy as np

from rxnrep.core.reaction import Reaction


class Transform:
    """
    Base class for transform.
    """

    def __init__(self, ratio: float):
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
    def __call__(self, reactants_g, products_g, reaction_g, reaction: Reaction):
        # select atoms to drop
        all_atoms = list(range(reaction.num_atoms))
        in_center = reaction.atoms_in_reaction_center
        not_in_center = [i for i in all_atoms if i not in in_center]
        n = int(self.ratio * len(not_in_center))
        atoms_to_drop = np.random.permutation(not_in_center)[:n]

        if not atoms_to_drop:
            return reactants_g, products_g, reaction_g, reaction
        else:
            atoms_to_keep = [i for i in all_atoms if i not in atoms_to_drop]

        # extract atom dgl subgraph
        g = reactants_g
        nodes = {k: list(range(g.num_nodes(k))) for k in g.ntypes}
        nodes["atom"] = atoms_to_keep
        sub_reactants_g = dgl.node_subgraph(g, nodes)

        g = products_g
        nodes = {k: list(range(g.num_nodes(k))) for k in g.ntypes}
        nodes["atom"] = atoms_to_keep
        sub_products_g = dgl.node_subgraph(g, nodes)

        return sub_reactants_g, sub_products_g, reactants_g, reaction
