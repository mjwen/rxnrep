import logging
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import dgl
import numpy as np
import torch
from sklearn.utils import class_weight

from rxnrep.data.dataset import BaseDataset
from rxnrep.data.grapher import (
    AtomTypeFeatureMasker,
    get_atom_distance_to_reaction_center,
    get_bond_distance_to_reaction_center,
)
from rxnrep.data.io import read_smiles_tsv_dataset

logger = logging.getLogger(__name__)


class USPTODataset(BaseDataset):
    """
    USPTO dataset.

    See the base class for docs of arguments not given here.

    Args:
        filename: tsv file of smiles reactions and labels
        max_hop_distance: maximum allowed hop distance from the reaction center for
            atom and bond. Used to determine atom and bond label. If `None`, not used.
        atom_type_masker_ratio: ratio of atoms whose atom type to be masked in each
            reaction. If `None`, not applied.
        atom_type_masker_use_masker_value: whether to use the calculate masker value. If
            `True`, use it, if `False` do not mask the atom features.
            Ignored if `atom_type_masker_ratio` is None.
            Note the difference between this and `atom_type_masker_ratio`; if
            `atom_type_masker_ratio` is `None`, the masker is not used in the sense
            that the masker is not even created. But `atom_type_masker_use_mask_value`
            be of `False` means that the masker are created, but we simply skip the
            change of the atom type features of the masked atoms.
    """

    def __init__(
        self,
        filename: Union[str, Path],
        atom_featurizer: Callable,
        bond_featurizer: Callable,
        global_featurizer: Callable,
        *,
        init_state_dict: Optional[Union[Dict, Path]] = None,
        transform_features: bool = True,
        return_index: bool = True,
        num_processes: int = 1,
        # args to control labels
        max_hop_distance: Optional[int] = None,
        atom_type_masker_ratio: Optional[float] = None,
        atom_type_masker_use_masker_value: Optional[bool] = None,
    ):

        super().__init__(
            filename,
            atom_featurizer,
            bond_featurizer,
            global_featurizer,
            init_state_dict=init_state_dict,
            transform_features=transform_features,
            return_index=return_index,
            num_processes=num_processes,
        )

        # labels and metadata (one inner dict for each reaction)
        self.labels = [{}] * len(self.reactions)
        self.medadata = [{}] * len(self.reactions)

        # atom bond hop distance label
        self.max_hop_distance = max_hop_distance

        # atom type maker label (this need special handling because we generate it
        # dynamically; different label is generated each time it is called)
        if atom_type_masker_ratio is None:
            self.atom_type_masker = None
        else:
            self.atom_type_masker = AtomTypeFeatureMasker(
                allowable_types=self._species,
                feature_name=self.feature_name["atom"],
                feature_mean=self._feature_mean["atom"],
                feature_std=self._feature_std["atom"],
                ratio=atom_type_masker_ratio,
                use_masker_value=atom_type_masker_use_masker_value,
            )

            # move the storage of atom features from the graph to self.atom_features
            self.atom_features = [
                {
                    "reactants": reactants_g.nodes["atom"].data.pop("feat"),
                    "products": products_g.nodes["atom"].data.pop("feat"),
                }
                for reactants_g, products_g, _ in self.dgl_graphs
            ]

        self.generate_labels()
        self.generate_metadata()

    def read_file(self, filename: Path):
        logger.info("Start reading dataset ...")

        succeed_reactions, failed = read_smiles_tsv_dataset(
            filename, remove_H=True, nprocs=self.nprocs
        )

        counter = Counter(failed)
        logger.info(
            f"Finish reading dataset. Number succeed {counter[False]}, "
            f"number failed {counter[True]}."
        )

        return succeed_reactions, failed

    def generate_labels(self):

        logger.info("Start generating labels ...")

        for i, rxn in enumerate(self.reactions):

            # atom bond hop distance label
            if self.max_hop_distance is not None:

                atom_hop = get_atom_distance_to_reaction_center(
                    rxn, max_hop=self.max_hop_distance
                )

                bond_hop = get_bond_distance_to_reaction_center(
                    rxn, atom_hop_distances=atom_hop, max_hop=self.max_hop_distance
                )
                self.labels[i].update(
                    {
                        "atom_hop_dist": torch.as_tensor(atom_hop, dtype=torch.int64),
                        "bond_hop_dist": torch.as_tensor(bond_hop, dtype=torch.int64),
                    }
                )

        logger.info("Finish generating labels ...")

    def generate_metadata(self):

        logger.info("Start generating metadata ...")

        for i, (rxn, label) in enumerate(zip(self.reactions, self.labels)):

            meta = {
                "reactant_num_molecules": len(rxn.reactants),
                "product_num_molecules": len(rxn.products),
                "num_unchanged_bonds": len(rxn.unchanged_bonds),
                "num_lost_bonds": len(rxn.lost_bonds),
                "num_added_bonds": len(rxn.added_bonds),
            }

            # add atom/bond hop dist to meta, which is used in hop dist pool
            if self.max_hop_distance is not None:
                meta.update(
                    {
                        "atom_hop_dist": label["atom_hop_dist"],
                        "bond_hop_dist": label["bond_hop_dist"],
                    }
                )

            self.medadata[i].update(meta)

        logger.info("Finish generating metadata ...")

    def get_class_weight(self):
        """
        Create class weight to be used in cross entropy losses.
        """
        return get_atom_bond_hop_dist_class_weight(self.labels, self.max_hop_distance)

    def __getitem__(self, item: int):
        """
        Get data point with index.
        """
        reactants_g, products_g, reaction_g = self.dgl_graphs[item]
        reaction = self.reactions[item]
        label = self.labels[item]
        meta = self.medadata[item]

        # atom type masker (this needs to be here since we generate it dynamically;
        # different label is generated each time it is called)
        if self.atom_type_masker is not None:
            atom_feats = self.atom_features[item]

            # Assign atom features bach to graph. Should clone to keep
            # self.atom_features intact.
            reactants_g.nodes["atom"].data["feat"] = (
                atom_feats["reactants"].clone().detach()
            )
            products_g.nodes["atom"].data["feat"] = (
                atom_feats["products"].clone().detach()
            )

            (
                reactants_g,
                products_g,
                is_masked,
                masked_labels,
            ) = self.atom_type_masker.mask_features(
                reactants_g,
                products_g,
                reaction,
            )

            # add info to label and meta
            label["masked_atom_type"] = torch.as_tensor(
                masked_labels, dtype=torch.int64
            )
            meta["is_atom_masked"] = torch.as_tensor(is_masked, dtype=torch.bool)

        if self.return_index:
            return item, reactants_g, products_g, reaction_g, meta, label
        else:
            return reactants_g, products_g, reaction_g, meta, label

    @staticmethod
    def collate_fn(samples):
        indices, reactants_g, products_g, reaction_g, metadata, labels = map(
            list, zip(*samples)
        )

        batched_indices = torch.as_tensor(indices)

        batched_molecule_graphs = dgl.batch(reactants_g + products_g)
        batched_reaction_graphs = dgl.batch(reaction_g, ndata=None, edata=None)

        # labels
        keys = labels[0].keys()
        batched_labels = {k: torch.cat([d[k] for d in labels]) for k in keys}

        # metadata used to split global and bond features
        keys = metadata[0].keys()
        batched_metadata = {k: [d[k] for d in metadata] for k in keys}

        # convert some metadata to tensor
        for k, v in batched_metadata.items():
            if k in ["atom_hop_dist", "bond_hop_dist", "is_atom_masked"]:
                # each element of v is a 1D tensor
                batched_metadata[k] = torch.cat(v)

        return (
            batched_indices,
            batched_molecule_graphs,
            batched_reaction_graphs,
            batched_labels,
            batched_metadata,
        )


class SchneiderDataset(USPTODataset):
    """
    Schneider 50k USPTO dataset with class labels for reactions.

    The difference between this and the USPTO dataset is that there is class label in
    this dataset and no class label in USPTO. This is added as the `reaction_class`
    in the `labels`.
    """

    def generate_labels(self):
        """
        Labels for all reactions.

        Add `reaction_class`.
        """
        super().generate_labels()

        for i, rxn in enumerate(self.reactions):
            rxn_class = rxn.get_property("label")
            self.labels[i]["reaction_class"] = torch.as_tensor(
                [int(rxn_class)], dtype=torch.int64
            )

    def get_class_weight(
        self, num_reaction_classes: int = 50, class_weight_as_1: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Create class weight to be used in cross entropy losses.

        Args:
            num_reaction_classes: number of reaction classes in the dataset. The class
            labels should be 0, 1, 2, ... num_reaction_classes-1.
            class_weight_as_1: If `True`, the weight for all classes is set to 1.0;
                otherwise, it is inversely proportional to the number of data points in
                the dataset
        """
        # class weight for atom hop and bond hop
        weight = super().get_class_weight()

        if class_weight_as_1:
            w = torch.ones(num_reaction_classes)
        else:
            rxn_classes = [rxn.get_property("label") for rxn in self.reactions]

            # class weight for reaction classes
            w = class_weight.compute_class_weight(
                "balanced",
                classes=list(range(num_reaction_classes)),
                y=rxn_classes,
            )
            w = torch.as_tensor(w, dtype=torch.float32)

        weight["reaction_class"] = w

        return weight


def get_atom_bond_hop_dist_class_weight(
    labels: List[Dict[str, torch.Tensor]],
    max_hop_distance: int,
    only_break_bond: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Create class weight to be used in cross entropy losses.

    Args:
        labels: a list of dict that contains `atom_hop_dist` and `bond_hop_dist` labels.
            See `get_atom_distance_to_reaction_center()` and
            `get_bond_distance_to_reaction_center` in grapher.py for how these labels
            are generated and what the mean.
        max_hop_distance: max hop distance used to generate the labels.
        only_break_bond: in all the reactions, there is only bond breaking, but not
            bond forming. If there is no bond forming, allowed number of classes for atom
            and bond hop distances are both different from reactions with bond forming.

    Returns:
        {name: weight}: class weight for atom/bond hop distance labels
    """

    if only_break_bond:
        atom_hop_num_classes = [max_hop_distance + 1]
        bond_hop_num_classes = max_hop_distance + 1
    else:
        atom_hop_num_classes = [max_hop_distance + 2, max_hop_distance + 3]
        bond_hop_num_classes = max_hop_distance + 2

    # Unique labels should be `list(range(atom_hop_num_classes))`.
    # The labels are:
    # atoms in lost bond: class 0
    # atoms in unchanged bond: class 1 to max_hop_distance
    # If there are added bonds
    # atoms in added bonds: class max_hop_distance + 1
    # If there are atoms associated with both lost and added bonds, the class label for
    # them is: max_hop_distance + 2
    all_atom_hop_labels = np.concatenate([lb["atom_hop_dist"] for lb in labels])

    unique_labels = sorted(set(all_atom_hop_labels))
    if unique_labels not in [list(range(a)) for a in atom_hop_num_classes]:
        raise RuntimeError(
            f"Unable to compute atom class weight; some classes do not have valid "
            f"labels. num_classes: {atom_hop_num_classes} unique labels: "
            f"{unique_labels}."
        )

    atom_hop_weight = class_weight.compute_class_weight(
        "balanced",
        classes=unique_labels,
        y=all_atom_hop_labels,
    )

    # Unique labels should be `list(range(bond_hop_num_classes))`.
    # The labels are:
    # lost bond: class 0
    # unchanged: class 1 to max_hop_distance
    # If there are added bonds:
    # add bonds: class max_hop_distance + 1
    all_bond_hop_labels = np.concatenate([lb["bond_hop_dist"] for lb in labels])

    unique_labels = sorted(set(all_bond_hop_labels))
    if unique_labels != list(range(bond_hop_num_classes)):
        raise RuntimeError(
            f"Unable to compute bond class weight; some classes do not have valid "
            f"labels. num_classes: {bond_hop_num_classes} unique labels: "
            f"{unique_labels}"
        )

    bond_hop_weight = class_weight.compute_class_weight(
        "balanced",
        classes=unique_labels,
        y=all_bond_hop_labels,
    )

    weight = {
        "atom_hop_dist": torch.as_tensor(atom_hop_weight, dtype=torch.float32),
        "bond_hop_dist": torch.as_tensor(bond_hop_weight, dtype=torch.float32),
    }

    return weight
