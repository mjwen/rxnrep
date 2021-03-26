from typing import Any, List, Tuple, Union

import dgl
import numpy as np
import torch

from rxnrep.core.reaction import Reaction


class Transform:
    """
    Base class for transform.

    Transforms will only be applied to atoms/bonds outside the reaction center.

    Args:
        select_mode: ['direct'/'ratio']. This determines how the number of augmented
            atoms/bonds are selected. if `direct`, ratio should be an integer, meaning
            the the number of atoms/bonds to augment. If `ratio`, ratio is the portion
            of atoms/bonds to augment. The portion multiplier is determined by
            ratio_multiplier. See below.
        ratio: the magnitude of augmentation.
        ratio_multiplier: [out_center|in_center]. the set of atoms/bonds used together
            with ratio to determine the number of atoms to augment. If `select_model =
            direct`, this is ignored and will selected ratio number of atoms/bonds
            outside the reaction center to augment. If `select_mode = ratio`:
            (1) ratio_multiplier = out_center, ratio*num_atoms/bonds_outside_center
            number of atoms/bonds will be augmented; (2) ratio_multiplier = in_center,
            ratio*num_atoms/bonds_in_center number of atoms/bonds will be augmented.
            Again, no matter what ratio_multiplier is, only atoms/bonds outside reaction
            center is augmented.
    """

    def __init__(
        self,
        ratio: Union[float, int],
        select_mode: str = "ratio",
        ratio_multiplier: str = "out_center",
    ):
        if select_mode == "direct":
            self.ratio = int(ratio)

        elif select_mode == "ratio":
            self.ratio = ratio

            supported = ["out_center", "in_center"]
            if ratio_multiplier not in supported:
                raise ValueError(
                    f"Expected ratio_multiplier be {supported}, got {ratio_multiplier}"
                )
            self.ratio_multiplier = ratio_multiplier

        else:
            supported = ["direct", "ratio"]
            if select_mode not in supported:
                raise ValueError(
                    f"Expected select_mode be {supported}, got {select_mode}"
                )

        self.select_mode = select_mode

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
        (not_in_center,) = np.nonzero(reaction.atom_distance_to_reaction_center)
        n = int(self.ratio * len(not_in_center))

        if n == 0:
            return reactants_g, products_g, reaction_g, None

        else:
            to_drop = np.random.choice(not_in_center, n, replace=False)
            to_keep = sorted(set(range(reaction.num_atoms)) - set(to_drop))

            # extract subgraph
            sub_reactants_g = get_node_subgraph(reactants_g, to_keep)
            sub_products_g = get_node_subgraph(products_g, to_keep)

            return sub_reactants_g, sub_products_g, reaction_g, None


class DropBond(Transform):
    """
    Transformation by dropping bonds. Atoms are NOT dropped.

    Do not drop bonds in reaction center.

    This assumes the bond is represented by two edges.
    """

    def __call__(self, reactants_g, products_g, reaction_g, reaction: Reaction):
        (not_in_center,) = np.nonzero(reaction.bond_distance_to_reaction_center)
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

            # extract dgl subgraph
            sub_reactants_g = get_edge_subgraph(reactants_g, reactant_bonds_to_keep)
            sub_products_g = get_edge_subgraph(products_g, product_bonds_to_keep)

            return sub_reactants_g, sub_products_g, reaction_g, None


class MaskAtomAttribute(Transform):
    """
    Only mask atoms not in the center.

    Args:
        ratio:
        mask_value: values to use for the masked features. The features of the masked
            bonds are set to this value. Since we normalize all the input features by
            subtracting mean and divide std, a value of 0 is equivalent to setting all
            the input features to mean, and a value of 1 is equivalent to setting all
            the input features to mean+std.
    """

    def __init__(
        self,
        ratio: float,
        select_mode: str = "ratio",
        ratio_multiplier: str = "out_center",
        mask_value: Union[float, torch.Tensor] = 0.0,
    ):
        super().__init__(ratio, select_mode, ratio_multiplier)
        self.mask_value = mask_value

    def __call__(self, reactants_g, products_g, reaction_g, reaction: Reaction):
        (not_in_center,) = np.nonzero(reaction.atom_distance_to_reaction_center)
        n = int(self.ratio * len(not_in_center))

        if n == 0:
            return reactants_g, products_g, reaction_g, None

        else:
            selected = sorted(np.random.choice(not_in_center, n, replace=False))

            # modify feature in-place
            reactants_g.nodes["atom"].data["feat"][selected] = self.mask_value
            products_g.nodes["atom"].data["feat"][selected] = self.mask_value

            return reactants_g, products_g, reaction_g, None


class MaskBondAttribute(Transform):
    """
    Only mask bonds not in the center.

    This will modify feature in place. So please backup the feature before calling this.

    Args:
        ratio:
        mask_value: values to use for the masked features. The features of the masked
            bonds are set to this value. Since we normalize all the input features by
            subtracting mean and divide std, a value of 0 is equivalent to setting all
            the input features to mean, and a value of 1 is equivalent to setting all
            the input features to mean+std.
    """

    def __init__(
        self,
        ratio: float,
        select_mode: str = "ratio",
        ratio_multiplier: str = "out_center",
        mask_value: Union[float, torch.Tensor] = 0.0,
    ):
        super().__init__(ratio, select_mode, ratio_multiplier)
        self.mask_value = mask_value

    def __call__(self, reactants_g, products_g, reaction_g, reaction: Reaction):
        (not_in_center,) = np.nonzero(reaction.bond_distance_to_reaction_center)
        n = int(self.ratio * len(not_in_center))

        if n == 0:
            return reactants_g, products_g, reaction_g, None

        else:
            indices = np.random.choice(not_in_center, n, replace=False)
            x = indices * 2  # each bond has two edges (2i and 2i+1)
            selected = sorted(np.concatenate((x, x + 1)))

            # modify feature in-place
            reactants_g.edges["bond"].data["feat"][selected] = self.mask_value
            products_g.edges["bond"].data["feat"][selected] = self.mask_value

            return reactants_g, products_g, reaction_g, None


class Subgraph(Transform):
    """
    Reaction center ego-subgraph, similar to appendix A2 of
    `Graph Contrastive Learning with Augmentations`, https://arxiv.org/abs/2010.13902.

    The difference is that we start with all atoms in reaction center, where as in the
    paper, it starts with a randomly chosen atom.
    """

    def __call__(self, reactants_g, products_g, reaction_g, reaction: Reaction):
        distance = np.asarray(reaction.atom_distance_to_reaction_center)
        in_center = np.argwhere(distance == 0).reshape(-1).tolist()

        # number of not in center atoms to sample
        num_in_center = len(in_center)
        num_not_in_center = len(distance) - num_in_center

        if self.select_mode == "direct":
            num_sample = self.ratio
        elif self.select_mode == "ratio":
            if self.ratio_multiplier == "out_center":
                num_sample = int(self.ratio * num_not_in_center)
            elif self.ratio_multiplier == "in_center":
                num_sample = int(self.ratio * num_in_center)
            else:
                raise ValueError
        else:
            raise ValueError
        num_sample = min(num_sample, num_not_in_center)

        if num_sample == 0:
            return reactants_g, products_g, reaction_g, None

        else:
            # Initialize subgraph as atoms in the center
            sub_graph = in_center

            # Initialize neighbors (do not contain atoms already in sub_graph)
            # It is sufficient to use the reactants graph to get the neighbors,
            # since beyond the reaction center, the reactants and products graphs are
            # the same.
            neigh = np.concatenate(
                [reactants_g.successors(n, etype="bond").numpy() for n in sub_graph]
            )

            # Given reaction graph,
            #
            #       C2---C3
            #      /      \
            # C0--C1      C4--C5
            #
            # suppose C1-C2 is a lost bond and C2-C3 is an added bond, the above
            # `neigh` will include C0, C1, C2, and C4, but NO C3, because we use
            # successors of reactants_g, and that C2-C3 is an added bond.
            # So, to get all neigh (including in_center atoms) the below union is needed.
            neigh = set(neigh).union(sub_graph).difference(sub_graph)

            while len(sub_graph) < num_in_center + num_sample:

                if len(neigh) == 0:  # e.g. H--H --> H + H, or all atoms included
                    break

                sample_atom = np.random.choice(list(neigh))

                assert (
                    sample_atom not in sub_graph
                ), "Something went wrong, this should not happen"

                sub_graph.append(sample_atom)
                neigh = neigh.union(
                    reactants_g.successors(sample_atom, etype="bond").numpy()
                )

                # remove subgraph atoms from neigh
                neigh = neigh.difference(sub_graph)

            # extract subgraph
            selected = sorted(sub_graph)
            sub_reactants_g = get_node_subgraph(reactants_g, selected)
            sub_products_g = get_node_subgraph(products_g, selected)

            return sub_reactants_g, sub_products_g, reaction_g, None


class SubgraphBFS(Transform):
    """
    Reaction center ego-subgraph, breadth first search.
    """

    def __call__(self, reactants_g, products_g, reaction_g, reaction: Reaction):
        distance = np.asarray(reaction.atom_distance_to_reaction_center)
        in_center = np.argwhere(distance == 0).reshape(-1).tolist()

        # number of not in center atoms to sample
        num_in_center = len(in_center)
        num_not_in_center = len(distance) - num_in_center

        if self.select_mode == "direct":
            num_sample = self.ratio
        elif self.select_mode == "ratio":
            if self.ratio_multiplier == "out_center":
                num_sample = int(self.ratio * num_not_in_center)
            elif self.ratio_multiplier == "in_center":
                num_sample = int(self.ratio * num_in_center)
            else:
                raise ValueError
        else:
            raise ValueError
        num_sample = min(num_sample, num_not_in_center)

        if num_sample == 0:
            return reactants_g, products_g, reaction_g, None

        else:
            num_target = num_in_center + num_sample

            # Initialize subgraph as atoms in the center (shell 0)
            sub_graph = in_center

            shell = 1
            while len(sub_graph) < num_target:
                candidate = np.argwhere(distance == shell).reshape(-1)
                assert (
                    candidate.size > 0
                ), "Cannot find candidate, this should not happen"

                if len(sub_graph) + len(candidate) > num_target:
                    selected = np.random.choice(
                        candidate, num_target - len(sub_graph), replace=False
                    )
                else:
                    selected = candidate
                sub_graph.extend(selected)

                shell += 1

            # extract subgraph
            selected = sorted(sub_graph)
            sub_reactants_g = get_node_subgraph(reactants_g, selected)
            sub_products_g = get_node_subgraph(products_g, selected)

            return sub_reactants_g, sub_products_g, reaction_g, None


class IdentityTransform(Transform):
    """
    Identity transform that does not modify the graph.
    """

    def __call__(
        self, reactants_g, products_g, reaction_g, reaction: Reaction
    ) -> Tuple[dgl.DGLGraph, dgl.DGLGraph, dgl.DGLGraph, Any]:

        return reactants_g, products_g, reaction_g, reaction


def get_edge_subgraph(g, edges: List[int], edge_type: str = "bond"):
    edges_dict = {k: list(range(g.num_edges(k))) for k in g.etypes}
    edges_dict[edge_type] = edges
    return dgl.edge_subgraph(g, edges_dict, preserve_nodes=True, store_ids=False)


def get_node_subgraph1(g, nodes: List[int], node_type: str = "atom"):
    """
    This cannot preserve relative order of edges.
    """
    nodes_dict = {k: list(range(g.num_nodes(k))) for k in g.ntypes}
    nodes_dict[node_type] = nodes
    return dgl.node_subgraph(g, nodes_dict, store_ids=False)


def get_node_subgraph(g, nodes: List[int], node_type: str = "atom") -> dgl.DGLGraph:
    """
    Node subgraph that preserves relative edge order.

    This has the same functionality as dgl.node_subgraph(); but this also preserves
    relative order of the edges.
    """

    # node mapping, old to new
    nodes = sorted(nodes)
    node_map = {n: i for i, n in enumerate(nodes)}

    edges_dict = {}
    edge_feats = {}
    for rel in g.canonical_etypes:
        srctype, etype, dsttype = rel

        u, v, eid = g.edges(form="all", order="eid", etype=rel)

        if srctype != node_type and dsttype != node_type:
            # neither is node type
            edges_dict[rel] = (u, v)
            edge_feats[rel] = g.edges[rel].data
        else:
            u = u.numpy()
            v = v.numpy()

            if srctype == node_type and dsttype == node_type:
                # both are node type
                isin = np.isin(u, nodes) * np.isin(v, nodes)
                new_u = [node_map[i] for i in u[isin]]
                new_v = [node_map[i] for i in v[isin]]
            elif srctype == node_type:
                # src is node type
                isin = np.isin(u, nodes)
                new_u = [node_map[i] for i in u[isin]]
                new_v = v[isin]
            else:
                # dst is node type
                isin = np.isin(v, nodes)
                new_u = u[isin]
                new_v = [node_map[i] for i in v[isin]]

            edges_dict[rel] = (new_u, new_v)
            edge_feats[rel] = {k: v[isin] for k, v in g.edges[rel].data.items()}

    num_nodes_dict = {t: g.num_nodes(t) for t in g.ntypes}
    num_nodes_dict[node_type] = len(nodes)

    # create graph
    new_g = dgl.heterograph(edges_dict, num_nodes_dict=num_nodes_dict)

    # edge features
    for k, v in edge_feats.items():
        new_g.edges[k].data.update(v)

    # nodes features
    for t in g.ntypes:
        feats = g.nodes[t].data

        # select features of retaining nodes
        if t == node_type:
            feats = {k: v[nodes] for k, v in feats.items()}

            # add node ids
            # feats["_ID"] = torch.as_tensor(nodes)

        new_g.nodes[t].data.update(feats)

    return new_g
