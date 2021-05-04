import json
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import KFold

from rxnrep.data.splitter import train_test_split


def split_regression_data(filename, num_folds=5):
    """
    Split dataset into equally nfold and use n-1 fold as training set and 1 fold as
    test set.
    Args:
        filename:
        num_folds:

    Returns:
        List of a tuple (train_filename, test_filename) of the k fold split.
    """
    with open(filename, "r") as f:
        data = json.load(f)

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=35)

    fold_filenames = []
    for i, (train_index, test_index) in enumerate(kf.split(data)):
        train = [data[j] for j in train_index]
        test = [data[j] for j in test_index]

        prefix = Path.cwd().joinpath(f"cv_fold{i}")
        if not prefix.exists():
            os.makedirs(prefix)

        train_fname = prefix.joinpath("train.tsv")
        with open(train_fname, "w") as f:
            json.dump(train, f)

        test_fname = prefix.joinpath("test.tsv")
        with open(test_fname, "w") as f:
            json.dump(test, f)

        fold_filenames.append((train_fname, test_fname))

    return fold_filenames


def split_classification_data(
    filename, trainset_size, testset_size_min, stratify_column, num_folds=5
):
    """
    Stratified split of data in a monte carlo way, i.e. draw samples randomly in each
    group.

    Args:
        trainset_size: number of datapoints of each group to enter in trainset.
        testset_size_min: at least this number in each group should go to testset.

    """

    df = pd.read_csv(filename, sep="\t")

    fold_filenames = []
    for i in range(num_folds):
        df1, df2 = train_test_split(
            df,
            ratio=None,
            size=trainset_size,
            test_min=testset_size_min,
            stratify_column=stratify_column,
            random_seed=i,
        )
        prefix = Path.cwd().joinpath(f"cv_fold{i}")
        if not prefix.exists():
            os.makedirs(prefix)

        train_fname = prefix.joinpath("train.tsv")
        df1.to_csv(train_fname, index=False, sep="\t")

        test_fname = prefix.joinpath("test.tsv")
        df2.to_csv(test_fname, index=False, sep="\t")

        fold_filenames.append((train_fname, test_fname))

    return fold_filenames


def cross_validate(
    args,
    ModelClass,
    load_dataset_fn,
    main_train_fn,
    mode="classification",
    stratify_column=None,
    fold=5,
    project="tmp-rxnrep",
):

    # split data

    # all data provided via trainset_filename
    if mode == "classification":
        filenames = split_classification_data(
            args.trainset_filename,
            args.trainset_size,
            args.testset_size_min,
            stratify_column=stratify_column,
            num_folds=fold,
        )
    elif mode == "regression":
        filenames = split_regression_data(args.trainset_filename, num_folds=fold)
    else:
        raise ValueError(f"Not supported cross validation mode {mode}")

    for k, (train_filename, val_filename) in enumerate(filenames):

        # modify dataset path
        args.trainset_filename = train_filename
        args.valset_filename = args.testset_filename = val_filename

        # dataset
        train_loader, val_loader, test_loader = load_dataset_fn(args)

        # model
        model = ModelClass(args)

        main_train_fn(
            args,
            model,
            train_loader,
            val_loader,
            test_loader,
            project=project,
            log_dir=train_filename.parent,
        )
