import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from rxnrep.data.splitter import train_test_split
from rxnrep.utils.io import to_path


def kfold_split(
    filename: Union[str, Path],
    save_dir: Union[str, Path] = ".",
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 35,
) -> List[Tuple[Path, Path]]:
    """
    Kfold split of a data file.

    This creates `n_splits` directory named cv_fold_0, cv_fold_1... in the path give by
    `save_dir`. In each directory, a train file (e.g. train.tsv) and a test file (e.g.
    test.tsv) are save in each cv_fold_<n>.

    Currently support `tsv`, `csv` and `json` files, and the file type are determined
    based on suffix of the input file.

    Args:
        filename: name of the data file to split.
        save_dir: path to create the split directories.
        n_splits:
        shuffle:
        random_state:

    Returns:
        A list of (train, test) files.
    """

    filename = to_path(filename)
    file_type = filename.suffix.strip(".")

    supported = ["tsv", "csv", "json"]
    if file_type not in supported:
        raise ValueError(f"Expect one of {supported} file type, got {file_type}")

    # read data
    if file_type == "json":
        with open(filename, "r") as f:
            data = json.load(f)
    elif file_type == "csv":
        data = pd.read_csv(filename, sep=",")
    elif file_type == "tsv":
        data = pd.read_csv(filename, sep="\t")
    else:
        raise ValueError

    # split
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    fold_filenames = []
    for i, (train_index, test_index) in enumerate(kf.split(data)):

        prefix = to_path(save_dir).joinpath(f"cv_fold_{i}")
        if not prefix.exists():
            os.makedirs(prefix)
        train_fname = prefix.joinpath("train." + file_type)
        test_fname = prefix.joinpath("test." + file_type)
        fold_filenames.append((train_fname, test_fname))

        if file_type == "json":
            train = [data[j] for j in train_index]
            test = [data[j] for j in test_index]

            with open(train_fname, "w") as f:
                json.dump(train, f)

            with open(test_fname, "w") as f:
                json.dump(test, f)

        elif file_type in ["tsv", "csv"]:
            train = data.iloc[train_index]
            test = data.iloc[test_index]

            sep = ","
            if file_type == "tsv":
                sep = "\t"

            train.to_csv(train_fname, index=False, sep=sep)
            test.to_csv(test_fname, index=False, sep=sep)

        else:
            raise ValueError

    return fold_filenames


def multi_train_test_split(
    filename: Union[str, Path],
    trainset_size: int,
    testset_size_min: int,
    stratify: str = None,
    save_dir: Union[str, Path] = ".",
    n_splits: int = 5,
    random_state: int = 35,
) -> List[Tuple[Path, Path]]:
    """
    Stratified train test split of data multiple times.

    Note, this is not folded split, i.e. each split is not related to the previous one.
    Here, each split is independent of the previous one.

    Args:
        filename: name of the data file to split.
        trainset_size: number of data points of each stratify group to enter trainset.
        testset_size_min: at least this number of data points in each stratify group
            should enter testset.
        stratify: column name of the tsv data where the dataset class (group) info is
            stored.
        save_dir: path to create the split directories.
        n_splits:
        random_state:

    Returns:
        A list of (train, test) files.
    """

    df = pd.read_csv(filename, sep="\t")

    fold_filenames = []
    for i in range(n_splits):
        df1, df2 = train_test_split(
            df,
            ratio=None,
            size=trainset_size,
            test_min=testset_size_min,
            stratify_column=stratify,
            random_seed=random_state + i,
        )

        prefix = to_path(save_dir).joinpath(f"cv_fold_{i}")
        if not prefix.exists():
            os.makedirs(prefix)

        train_fname = prefix.joinpath("train.tsv")
        df1.to_csv(train_fname, index=False, sep="\t")

        test_fname = prefix.joinpath("test.tsv")
        df2.to_csv(test_fname, index=False, sep="\t")

        fold_filenames.append((train_fname, test_fname))

    return fold_filenames


def read_splits(filename: Union[str, Path]) -> List[Tuple[Path, Path]]:
    """
    Read already splitted files from directory.

    Directory should have the below structure:


    path:
        - cv_fold_1
            - train.tsv
            - test.tsv
        - cv_fold_2
            - train.tsv
            - test.tsv
        - cv_fold_3
            - train.tsv
            - test.tsv
        ...

    Args:
        filename: path to the directory containing the files. This is bad use of
            argument `filename`. It should be `path` indeed. We name it `filename` to
            be consistent with the other split functions for the easiness of use: in
            running script, we update `filename` from datamodule.

    Returns:
        A list of (train, test) files.
    """
    path = to_path(filename)

    fold_names = []
    for p in sorted(path.glob("cv_fold_*")):
        train = p.joinpath("train.tsv")
        test = p.joinpath("test.tsv")

        if not train.exists():
            raise RuntimeError(f"Cannot find `train.tsv` in {p}")
        if not test.exists():
            raise RuntimeError(f"Cannot find `test.tsv` in {p}")

        fold_names.append((train, test))

    if not fold_names:
        raise RuntimeError(f"No data files found at {path}")

    return fold_names


def compute_metric_statistics(data: List[Dict[str, float]]):
    """
    Compute statistics of multiple properties given in a list of dict.
    """
    aggregated = defaultdict(list)  # type: Dict[str, List[float]]
    for d in data:
        for k, v in d.items():
            aggregated[k].append(v)

    mean = {k: np.mean(v) for k, v in aggregated.items()}
    std = {k: np.std(v) for k, v in aggregated.items()}

    return dict(aggregated), mean, std
