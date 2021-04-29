from collections import defaultdict
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from rxnrep.data.dataset import BaseDatasetWithLabels


def train_test_split(
    df: pd.DataFrame,
    ratio: float = None,
    size: int = None,
    test_min: int = None,
    stratify_column: str = "reaction_type",
    random_seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a pandas dataframe into two by column according to ratio or size.

    Args:
        ratio: each group (determined by stratify_column) with a `ratio`
            portion of data goes to trainset and the remaining to testset.
            `ratio` and `size` can only have one to be true.
        size: each group (determined by stratify_column) with a fixed
            number of `size` data points go to trainset and the remaining to testset.
        test_min: at least this number of data points in each group should go to test
            set. If None, this is not used.

    Returns:
        part1: a dataframe containing ratio/size of data points of each group
        part2: a dataframe containing the remaining data points of each group
    """
    assert not (
        ratio is None and size is None
    ), "One of `ratio` or `size` should be provided"
    assert not (
        ratio is not None and size is not None
    ), "Only one of `ratio` or `size` should be not None"

    if random_seed is not None:
        np.random.seed(random_seed)

    grouped_df = df.groupby(by=stratify_column)

    train = []
    test = []
    for _, group in grouped_df:
        n = len(group)

        if ratio is not None:
            n_train = int(n * ratio)
        else:
            n_train = size

        # adjust to ensure at least test_min goes to test set
        if test_min is not None:
            if n - n_train < test_min:
                n_train = max(0, n - test_min)

        indices = np.random.permutation(n)
        part1_indices = indices[:n_train]
        part2_indices = indices[n_train:]

        train.append(group.iloc[part1_indices])
        test.append(group.iloc[part2_indices])

    train = pd.concat(train)
    test = pd.concat(test)

    return train, test


def train_validation_test_split(dataset, validation=0.1, test=0.1, random_seed=None):
    """
    Split a dataset into training, validation, and test set.

    The training set will be automatically determined based on `validation` and `test`,
    i.e. train = 1 - validation - test.

    Args:
        dataset: the dataset
        validation (float, optional): The amount of data (fraction) to be assigned to
            validation set. Defaults to 0.1.
        test (float, optional): The amount of data (fraction) to be assigned to test
            set. Defaults to 0.1.
        random_seed (int, optional): random seed that determines the permutation of the
            dataset. Defaults to 35.

    Returns:
        [train set, validation set, test_set]
    """
    assert validation + test < 1.0, "validation + test >= 1"
    size = len(dataset)
    num_val = int(size * validation)
    num_test = int(size * test)
    num_train = size - num_val - num_test

    if random_seed is not None:
        np.random.seed(random_seed)
    idx = np.random.permutation(size)
    train_idx = idx[:num_train]
    val_idx = idx[num_train : num_train + num_val]
    test_idx = idx[num_train + num_val :]
    return [
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    ]


def train_validation_test_split_test_with_all_bonds_of_mol(
    dataset, validation=0.1, test=0.1, random_seed=None
):
    """
    Split a dataset into training, validation, and test set.

    Different from `train_validation_test_split`, where the split of dataset is bond
    based, here the bonds from a molecule either goes to (train, validation) set or
    test set. This is used to evaluate the prediction order of bond energy.

    The training set will be automatically determined based on `validation` and `test`,
    i.e. train = 1 - validation - test.

    Args:
        dataset: the dataset
        validation (float, optional): The amount of data (fraction) to be assigned to
            validation set. Defaults to 0.1.
        test (float, optional): The amount of data (fraction) to be assigned to test
            set. Defaults to 0.1.
        random_seed (int, optional): random seed that determines the permutation of the
            dataset. Defaults to 35.

    Returns:
        [train set, validation set, test_set]
    """
    assert validation + test < 1.0, "validation + test >= 1"
    size = len(dataset)
    num_val = int(size * validation)
    num_test = int(size * test)
    num_train = size - num_val - num_test

    # group by molecule
    groups = defaultdict(list)
    for i, (_, label) in enumerate(dataset):
        groups[label["id"]].append(i)
    groups = [val for key, val in groups.items()]

    # permute on the molecule level
    if random_seed is not None:
        np.random.seed(random_seed)
    idx = np.random.permutation(len(groups))
    test_idx = []
    train_val_idx = []
    for i in idx:
        if len(test_idx) < num_test:
            test_idx.extend(groups[i])
        else:
            train_val_idx.extend(groups[i])

    # permute on the bond level for train and validation
    idx = np.random.permutation(train_val_idx)
    train_idx = idx[:num_train]
    val_idx = idx[num_train:]

    return [
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    ]


def train_validation_test_split_selected_bond_in_train(
    dataset, validation=0.1, test=0.1, random_seed=None, selected_bond_type=None
):
    """
    Split a dataset into training, validation, and test set.

    The training set will be automatically determined based on `validation` and `test`,
    i.e. train = 1 - validation - test.

    Args:
        dataset: the dataset
        validation (float, optional): The amount of data (fraction) to be assigned to
            validation set. Defaults to 0.1.
        test (float, optional): The amount of data (fraction) to be assigned to test
            set. Defaults to 0.1.
        random_seed (int, optional): random seed that determines the permutation of the
            dataset. Defaults to 35.
        selected_bond_type (tuple): breaking bond in `selected_bond_type` are all
            included in training set, e.g. `selected_bonds = (('H','H'), (('H', 'F'))`

    Returns:
        [train set, validation set, test_set]
    """
    assert validation + test < 1.0, "validation + test >= 1"
    size = len(dataset)
    num_val = int(size * validation)
    num_test = int(size * test)
    # num_train = size - num_val - num_test

    # index of bond in selected_bond
    selected_idx = []
    selected = [tuple(sorted(i)) for i in selected_bond_type]
    for i, (_, _, label) in enumerate(dataset):
        bond_type = tuple(sorted(label["id"].split("-")[-2:]))
        if bond_type in selected:
            selected_idx.append(i)

    all_idx = np.arange(size)
    all_but_selected_idx = list(set(all_idx) - set(selected_idx))

    if random_seed is not None:
        np.random.seed(random_seed)
    idx = np.random.permutation(all_but_selected_idx)

    val_idx = idx[:num_val]
    test_idx = idx[num_val : num_val + num_test]
    train_idx = list(idx[num_val + num_test :]) + selected_idx

    return [
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    ]


class Subset(BaseDatasetWithLabels):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    @property
    def feature_size(self):
        return self.dataset.feature_size

    @property
    def feature_name(self):
        return self.dataset.feature_name

    def __getitem__(self, item):
        return self.dataset[self.indices[item]]

    def __len__(self):
        return len(self.indices)
