import logging
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import dgl
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler as sk_StandardScaler

logger = logging.getLogger(__name__)


class Transformer:
    """
    Base class for normalize the dataset, either the features for the labels.
    """

    def __init__(self):
        self._mean = None
        self._std = None

    def transform(self, data):
        """
        Transform the data, i.e. subtract mean and divide by standard deviation.
        """
        raise NotImplementedError

    def inverse_transform(self, data):
        """
        Inverse transform the data, i.e. multiply standard deviation and and add mean.
        """
        raise NotImplementedError

    def state_dict(self):
        d = {"mean": self._mean, "std": self._std}
        return d

    def load_state_dict(self, d: Dict[str, Any]):
        self._mean = d["mean"]
        self._std = d["std"]


class StandardScaler(Transformer):
    """
    A wrapper over `sklearn.preprocessing.StandardScaler`.
    """

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            data: 2D tensor (n_samples, n_features) where each feature is standardized.

        Returns:
            2D tensor with each feature standardized.
        """

        if self._mean is None or self._std is None:

            _, mean, std = _transform(data.numpy())

            # We do not use the returned standardrized data (_) because _transform
            # internally uses float64. We normalize it manually below to ensure the
            # same precision for training, validation, and test sets, because we may
            # use the mean and std obtained from training set and use it for validation
            # and tests sets.
            self._mean = torch.from_numpy(mean)
            self._std = torch.from_numpy(std)

        data = (data - self._mean) / self._std

        return data

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        assert (
            self._mean is not None and self._std is not None
        ), "Cannot do inverse transform; mean or std is `None`."

        data = data * self._std + self._mean

        return data


class GraphFeatureTransformer(Transformer):
    """
    Standardize graph features (both node and edge features) and place them back to
    the graph feature dict.

    Args:
        key: name of the feature in the graph features dict to transform.
    """

    def __init__(self, key: str = "feat"):
        super().__init__()
        self.key = key

    def transform(self, graphs: List[dgl.DGLGraph]) -> List[dgl.DGLGraph]:
        g = graphs[0]
        node_types = g.ntypes
        edge_types = g.etypes

        all_feats = {"node": defaultdict(list), "edge": defaultdict(list)}
        all_feats_size = {"node": defaultdict(list), "edge": defaultdict(list)}

        # obtain feats from graphs
        for g in graphs:
            for t in node_types:
                data = g.nodes[t].data.get(self.key, None)
                if data is not None:
                    all_feats["node"][t].append(data)
                    all_feats_size["node"][t].append(len(data))

            for t in edge_types:
                data = g.edges[t].data.get(self.key, None)
                if data is not None:
                    all_feats["edge"][t].append(data)
                    all_feats_size["edge"][t].append(len(data))

        # standardize
        if self._mean is None or self._std is None:

            self._std = {"node": {}, "edge": {}}
            self._mean = {"node": {}, "edge": {}}

            # We do not use the returned standardrized data (_) because _transform
            # internally uses float64. We normalize it manually below to ensure the
            # same precision for training, validation, and test sets, because we may
            # use the mean and std obtained from training set and use it for validation
            # and tests sets.
            for name in all_feats:
                for t, feats in all_feats[name].items():
                    _, mean, std = _transform(torch.cat(feats).numpy(), copy=False)
                    self._mean[name][t] = torch.from_numpy(mean)
                    self._std[name][t] = torch.from_numpy(std)

        # normalize
        for name in all_feats:
            for t, feats in all_feats[name].items():
                m = self._mean[name][t]
                s = self._std[name][t]
                feats = (torch.cat(feats) - m) / s
                all_feats[name][t] = feats

        # assign data back to graph
        name = "node"
        for t, feats in all_feats[name].items():
            feats = torch.split(feats, all_feats_size[name][t])
            for g, ft in zip(graphs, feats):
                g.nodes[t].data[self.key] = ft

        name = "edge"
        for t, feats in all_feats[name].items():
            feats = torch.split(feats, all_feats_size[name][t])
            for g, ft in zip(graphs, feats):
                g.edges[t].data[self.key] = ft

        return graphs


def _transform(
    data: np.ndarray, copy: bool = True, with_mean=True, with_std=True, threshold=1.0e-3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize data using `sklearn.preprocessing.StandardScaler`.

    Args:
        data: array to standardize
        copy: whether to copy the array

    Returns:
        rst: 2D array
        mean: 1D array
        std: 1D array
    """
    scaler = sk_StandardScaler(copy=copy, with_mean=with_mean, with_std=with_std)
    rst = scaler.fit_transform(data)
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

    # mean and std can be of different dtype as data, manually convert it back
    dtype = data.dtype
    rst = rst.astype(dtype, copy=False)
    mean = mean.astype(dtype, copy=False)
    std = std.astype(dtype, copy=False)

    return rst, mean, std
