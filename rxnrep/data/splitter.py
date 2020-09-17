import logging
import numpy as np
from collections import defaultdict
from rxnrep.data.dataset import Subset

logger = logging.getLogger(__name__)


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
