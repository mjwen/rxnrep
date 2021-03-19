from typing import Any, List, Tuple

import dgl
import numpy as np

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

        all_to_keep = []
        all_num_atoms = []

        start = 0
        for rxn in reactions:

            # select atoms to drop
            (not_in_center,) = np.nonzero(rxn.atom_distance_to_reaction_center)
            num_to_drop = int(self.ratio * len(not_in_center))
            to_drop = np.random.choice(not_in_center, num_to_drop, replace=False)

            # select atoms to keep (+start to convert local atom id to batched id)
            num_atoms = rxn.num_atoms
            to_keep = sorted(set(range(num_atoms)).difference(to_drop))
            to_keep = np.asarray(to_keep) + start

            all_to_keep.append(to_keep)
            all_num_atoms.append(num_atoms)

            start += num_atoms

        all_to_keep = np.concatenate(all_to_keep).tolist()

        # extract subgraph
        sub_reactants_g = get_node_subgraph(reactants_g, all_to_keep)
        sub_products_g = get_node_subgraph(products_g, all_to_keep)

        return sub_reactants_g, sub_products_g, reactions_g, None
