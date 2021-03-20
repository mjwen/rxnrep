from typing import Any, List, Set, Tuple

import dgl
import numpy as np
import torch

from rxnrep.core.reaction import Reaction
from rxnrep.data.transforms import get_node_subgraph


class TransformBatch:
    """
    Base class for transform a batch of graphs.
    """

    def __init__(self, ratio: float):
        assert 0 < ratio < 1, f"expect ratio be 0<ratio<1, got {ratio}"
        self.ratio = ratio

    def __call__(
        self, reactants_g, products_g, reactions_g, reactions: List[Reaction]
    ) -> Tuple[dgl.DGLGraph, dgl.DGLGraph, dgl.DGLGraph, Any]:
        """
        Args:
            reactants_g: dgl batched graph
            products_g: dgl batched graph
            reactions_g: dgl batched graph
            reactions:
        Returns:
            augmented_reactants_g:
            agumented_products_g:
            agumented_reactions_g:
            metadata:
        """
        raise NotImplementedError


class DropAtomBatch(TransformBatch):
    """
    Transformation by dropping atoms. The bonds of the atoms are also dropped.

    Atoms in reaction center are not dropped.

    This assumes the atom is a node.
    """

    def __call__(self, reactants_g, products_g, reactions_g, reactions: List[Reaction]):

        rct_batch_num_edges = {
            t: reactants_g.batch_num_edges(t).clone()
            for t in reactants_g.canonical_etypes
        }
        prdt_batch_num_edges = {
            t: products_g.batch_num_edges(t).clone()
            for t in products_g.canonical_etypes
        }

        all_to_keep = []
        all_num_atoms = []

        start = 0
        for i, rxn in enumerate(reactions):

            # select atoms to drop
            (not_in_center,) = np.nonzero(rxn.atom_distance_to_reaction_center)
            num_to_drop = int(self.ratio * len(not_in_center))
            to_drop = np.random.choice(not_in_center, num_to_drop, replace=False)

            # select atoms to keep (+start to convert local atom id to batched id)
            num_atoms = rxn.num_atoms
            to_keep = sorted(set(range(num_atoms)).difference(to_drop))
            to_keep = np.asarray(to_keep) + start
            all_to_keep.append(to_keep)

            # compute number of edges after dropping
            to_drop = to_drop + start
            drop_edge_pairs = get_edges_associated_with_nodes(
                reactants_g, to_drop, etype="bond"
            )
            update_batch_num_edge(i, rct_batch_num_edges, drop_edge_pairs, to_drop)

            # do not need to call get_edges_associated_with_nodes for products,
            # since the same number edges are dropped
            update_batch_num_edge(i, prdt_batch_num_edges, drop_edge_pairs, to_drop)

            all_num_atoms.append(num_atoms - len(to_drop))

            start += num_atoms

        all_to_keep = np.concatenate(all_to_keep).tolist()

        # extract subgraph
        sub_reactants_g = get_node_subgraph(reactants_g, all_to_keep)
        sub_products_g = get_node_subgraph(products_g, all_to_keep)

        # set batch info
        sub_reactants_g.set_batch_num_edges(rct_batch_num_edges)
        sub_reactants_g.set_batch_num_nodes(
            {
                "atom": torch.tensor(all_num_atoms),
                "global": reactants_g.batch_num_nodes("global"),
            }
        )

        sub_products_g.set_batch_num_edges(prdt_batch_num_edges)
        sub_products_g.set_batch_num_nodes(
            {
                "atom": torch.tensor(all_num_atoms),
                "global": products_g.batch_num_nodes("global"),
            }
        )

        return sub_reactants_g, sub_products_g, reactions_g, None


def get_edges_associated_with_nodes(
    g: dgl.DGLGraph, nodes: List[int], etype: str
) -> Set[Tuple[int, int]]:
    """
    Get all the edges (both inbound and outbound) that are connected to the set of
    given nodes.

    Args:
        g:
        nodes: the set of nodes
        etype: the type of edges to look at.

    Returns:
        Edges associated with give nodes. Each edge is given as a tuple the ids of the
        nodes that the edge connects to.
    """

    edge_pairs = []
    for n in nodes:
        pre = g.predecessors(n, etype=etype)
        suc = g.successors(n, etype=etype)

        pre_n = torch.stack((pre, torch.tensor([n] * len(pre), dtype=pre.dtype)), dim=1)
        n_suc = torch.stack((torch.tensor([n] * len(suc), dtype=suc.dtype), suc), dim=1)

        edge_pairs.extend([pre_n, n_suc])  # inbound and outbound

    edge_pairs = torch.cat(edge_pairs).numpy()

    # convert to a set of tuples; use set to remove repeated ones
    edge_pairs = set(map(tuple, edge_pairs))

    return edge_pairs


def update_batch_num_edge(i, batch_num_edges, drop_edge_pairs, to_drop):
    a2a = ("atom", "bond", "atom")
    a2g = ("atom", "a2g", "global")
    g2a = ("global", "g2a", "atom")
    batch_num_edges[a2a][i] = batch_num_edges[a2a][i] - len(drop_edge_pairs)
    batch_num_edges[a2g][i] = batch_num_edges[a2g][i] - len(to_drop)
    batch_num_edges[g2a][i] = batch_num_edges[g2a][i] - len(to_drop)
