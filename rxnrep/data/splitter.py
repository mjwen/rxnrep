from typing import Optional, Tuple

import numpy as np
import pandas as pd


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
        stratify_column: name of column used as label to do stratified split.
        random_seed:

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
