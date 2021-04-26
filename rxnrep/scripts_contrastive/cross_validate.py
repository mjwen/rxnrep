import os
from pathlib import Path

import pandas as pd

from rxnrep.data.splitter import split_df_into_two


def split_data(filename, data_column_name, fold=5):
    df = pd.read_csv(filename, sep="\t")

    fold_filenames = []
    for i in range(fold):
        df1, df2 = split_df_into_two(
            df, ratio=1 / fold, column_name=data_column_name, random_seed=i
        )
        prefix = Path.cwd().joinpath("cv_working_dir", f"fold{i}")
        if not prefix.exists():
            os.makedirs(prefix)

        train_fname = prefix.joinpath("train.tsv")
        val_fname = prefix.joinpath("val.tsv")

        df1.to_csv(val_fname, index=False, sep="\t")
        df2.to_csv(train_fname, index=False, sep="\t")

        fold_filenames.append((train_fname, val_fname))

    return fold_filenames


def cross_validate(
    args,
    ModelClass,
    load_dataset_fn,
    main_train_fn,
    data_column_name,
    fold=5,
    project="tmp-rxnrep",
):

    # split data
    # all data provided via trainset_filename
    filenames = split_data(args.trainset_filename, data_column_name, fold)

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
