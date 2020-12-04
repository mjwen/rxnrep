import logging
from collections import defaultdict
from typing import Dict, List, Optional

import dgl
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler as sk_StandardScaler

logger = logging.getLogger(__name__)


class StandardScaler:
    """
    Standardize features using `sklearn.preprocessing.StandardScaler`.

    Args:
        mean: 1D array of the mean
        std: 1D array of the standard deviation
    """

    def __init__(
        self, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None
    ):
        self._mean = mean
        self._std = std

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: 2D array each column is standardized.

        Returns:
            2D array with each column standardized.
        """

        if self._mean is not None and self._std is not None:
            X = (X - self._mean) / self._std
        else:
            X, self._mean, self._std = _transform(X, copy=True)

        return X


class HeteroGraphFeatureStandardScaler:
    """
    Standardize hetero graph features by centering and normalization.
    Only node features are standardized.

    The mean and std can be provided for standardization. If `None` they are computed
    from the features of the graphs.

    Args:
        mean: with node type as key and the mean value as the value
        std: with node type as key and the std value as the value
        key: key of the feature in the graph nodes to scale.

    Returns:
        Graphs with their node features standardized. Note, these graphs are the same
        as the input graphs. That means the features of the graphs are updated inplace.
    """

    def __init__(
        self,
        mean: Optional[Dict[str, torch.Tensor]] = None,
        std: Optional[Dict[str, torch.Tensor]] = None,
        key: Optional[str] = "feat",
    ):
        self._mean = mean
        self._std = std
        self.key = key

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    def __call__(self, graphs) -> List[dgl.DGLGraph]:
        g = graphs[0]
        node_types = g.ntypes
        node_feats = defaultdict(list)
        node_feats_size = defaultdict(list)

        # obtain feats from graphs
        for g in graphs:
            for nt in node_types:
                data = g.nodes[nt].data[self.key]
                node_feats[nt].append(data)
                node_feats_size[nt].append(len(data))

        dtype = node_feats[node_types[0]][0].dtype

        # standardize
        if self._mean is not None and self._std is not None:
            for nt in node_types:
                feats = (torch.cat(node_feats[nt]) - self._mean[nt]) / self._std[nt]
                node_feats[nt] = feats
        else:
            self._std = {}
            self._mean = {}
            for nt in node_types:
                X = torch.cat(node_feats[nt]).numpy()
                feats, mean, std = _transform(X, copy=False)
                node_feats[nt] = torch.as_tensor(feats, dtype=dtype)
                self._mean[nt] = torch.as_tensor(mean, dtype=dtype)
                self._std[nt] = torch.as_tensor(std, dtype=dtype)

        # assign data back to graph
        for nt in node_types:
            feats = torch.split(node_feats[nt], node_feats_size[nt])
            for g, ft in zip(graphs, feats):
                g.nodes[nt].data[self.key] = ft

        return graphs


def _transform(
    X: np.ndarray, copy: bool, with_mean=True, with_std=True, threshold=1.0e-3
):
    """
    Standardize X (subtract mean and divide by standard deviation) using
    `sklearn.preprocessing.StandardScaler`.

    Args:
        X: array to standardize
        copy: whether to copy the array

    Returns:
        rst: 2D array
        mean: 1D array
        std: 1D array
    """
    scaler = sk_StandardScaler(copy=copy, with_mean=with_mean, with_std=with_std)
    rst = scaler.fit_transform(X)
    mean = scaler.mean_
    std = scaler.scale_

    # check small standard deviation (we do not use scaler.scale_ because small values,
    # e.g. 0.0 are replaced by 1)
    sqrt_var = np.sqrt(scaler.var_)
    for i, v in enumerate(sqrt_var):
        if v <= threshold:
            logger.warning(
                "Standard deviation for feature {} is {}, smaller than {}. "
                "You may want to exclude this feature.".format(i, v, threshold)
            )

    return rst, mean, std
