from typing import List, Optional, Sequence, Tuple, Union

import dgl
import numpy as np
import torch

from rxnrep.core.reaction import Reaction


class AtomTypeFeatureMasker:
    """
    A class to mask atom type features.

    We mask the atom type (features) of certain atoms and let the NN to predict the
    atom type based on its neighboring info so as to require the NN to learn and
    retain atom type info in the representational vector.

    If feature mean and std are provided, the mask features are set to
    `-mean[start_idx:end_idx]/std[start_idx:end_idx]`, where start_idx and end_idx are
    the starting index and ending index of atom type features in the feature_mean and
    std sequences. This is equivalent to setting all one hot encoding digits to 0 when
    getting the features.
    If feature mean or std is not provided, the mask values are set to zeros.

    Args:
        allowable_types: all allowed atom types. The class labels for masked atoms are
            generated from this.
        feature_name: name of the atom features e.g. `atom type`, `in ring`....
            This is used to find the indices of the atom features stored in the graphs.
        feature_mean: 1D tensor. Mean of the atom features. This is used together with
            `feature_std`, to determine the feature values for masked atoms.
        feature_std: 1D tensor. Standard deviation of atom features.
        ratio: ratio of atoms in each reaction to mask.
        use_masker_value: if `True` the atom type features of the masked atoms are
            set to self.masker_value (mean/std or zero). If `False`, do not change
            the atom type features of the masked atoms; the hope is that by
            directly predicting the atom type with atom type features, the atom
            type information is retained in the final representation.
    """

    def __init__(
        self,
        allowable_types: List[str],
        feature_name: List[str],
        feature_mean: Optional[Union[Sequence, torch.Tensor]] = None,
        feature_std: Optional[Union[Sequence, torch.Tensor]] = None,
        ratio: float = 0.2,
        use_masker_value: bool = True,
    ):
        self.allowable_types = sorted(allowable_types)
        self.ratio = ratio
        self.use_masker_value = use_masker_value

        self.class_labels_map = {s: i for i, s in enumerate(self.allowable_types)}

        # indices of atom type features
        self.start_index = feature_name.index("atom_type_one_hot")
        self.end_index = len(feature_name) - feature_name[::-1].index(
            "atom_type_one_hot"
        )

        if feature_mean is not None and feature_std is not None:
            # set mask values to be - mean/std. This is equivalent to set all values of
            # the one hot encoding to 0
            mean = torch.as_tensor(feature_mean, dtype=torch.float32)
            std = torch.as_tensor(feature_std, dtype=torch.float32)
            mean = mean[self.start_index : self.end_index]
            std = std[self.start_index : self.end_index]
            self.mask_values = -mean / std
        else:
            self.mask_values = torch.zeros(self.end_index - self.start_index)

    def mask_features(
        self,
        reactants_g: dgl.DGLGraph,
        products_g: dgl.DGLGraph,
        reaction: Reaction,
    ) -> Tuple[dgl.DGLGraph, dgl.DGLGraph, List[bool], List[int]]:
        """
        Mask the atom type features.

        Args:
            reactants_g: reactants graph
            products_g: products graph
            reaction: the reaction

        Returns:
            reactants_g: the reactants graph with atom type feature updated
            products_g: the products graph with atom type feature updated
            is_atom_masked: whether an atom is masked or not. Should have a length of
                number of atoms in the reaction.
            masked_atom_labels: class label for the masked atoms. Should have a length
                of number of masked atoms as indicated in `masked`. The order of the
                labels are in correspondence with `is_atom_masked`. For example, suppose
                `is_atoms_masked = [True, False, True, False, False]` (i.e. atoms 0 and
                2 are masked), then `masked_atom_labels = [1, 5]` means the label for
                atom 0 is 1 and for atom 2 is 5.
        """
        num_atoms = sum([m.num_atoms for m in reaction.reactants])
        permuted_atoms = np.random.permutation(num_atoms).tolist()
        masked_atoms = permuted_atoms[: int(num_atoms * self.ratio)]

        # mask at least 1 atom for small molecules
        if not masked_atoms:
            masked_atoms = [permuted_atoms[0]]
        else:
            masked_atoms = sorted(masked_atoms)

        is_atom_masked = [
            True if i in masked_atoms else False for i in range(num_atoms)
        ]

        # set masked atom labels
        masked_atom_labels = [
            self.class_labels_map[reaction.species[i]] for i in masked_atoms
        ]

        if self.use_masker_value:
            # set atom features to masked values
            reactants_g.nodes["atom"].data["feat"][
                masked_atoms, self.start_index : self.end_index
            ] = self.mask_values

            products_g.nodes["atom"].data["feat"][
                masked_atoms, self.start_index : self.end_index
            ] = self.mask_values

        return reactants_g, products_g, is_atom_masked, masked_atom_labels
