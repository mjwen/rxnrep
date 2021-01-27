"""
Distributed and serial K-means clustering methods.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from pytorch_lightning.utilities import move_data_to_device
from scipy.sparse import csr_matrix
from tqdm import tqdm


class DistributedReactionCluster:
    """
    Distributed K-mean clustering, with multiple cluster heads.

    Adapted from deepcluster-v2 at:
    https://github.com/facebookresearch/swav/blob/master/main_deepclusterv2.py

    Args:
        num_centroids: the number of centroids in each cluster head. For example,
            `[K1, K2, K3]` means three cluster heads are used, with K1, K2, and K3
            number of centroids, respectively.
    """

    def __init__(
        self,
        model,
        data_loader,
        num_centroids: List[int] = (10, 10),
        device=None,
    ):
        self.model = model
        self.data_loader = data_loader
        self.num_centroids = num_centroids
        self.device = device

        self.local_data = None
        self.local_index = None

    def get_cluster_assignments(
        self,
        num_iters: int = 10,
        similarity: str = "cosine",
        centroids: Union[List[torch.Tensor], str, None] = "random",
        predict_only: bool = False,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Get the assignments of the data points to the clusters.

        Note, there are multiple cluster heads, determined by `num_centroids`.

        Args:
            num_iters: number of iterations for k-means
            similarity: [`cosine`|`l2`]. similarity measure of the distance between
                data and centroid.
            centroids: initial centroids of the clusters. This shape of each tensor
                should correspond to `num_centroids`. For example, if `num_centroids` is
                `(K1, K2, K3)`, then `centroids` should be a list of 3 tensors, with shape
                (K1, D), (K2, D), and (K3, D), respectively.
                If `random`, will randomly select centroids from the data.
            predict_only: only predict the cluster assignments, without updating
                centroids. If `True`, `centroids_init` should be given as a list of
                tensor.

        Returns:
            assignments: the assignments of the data points to the clusters. Each element
                of the list gives the assignment for one clustering head. It is a
                tensor of shape (N_global,), where N_global is the number of total data
                points. Returned tensors will be on cpu.
            centroids: Centroids of the clusters. This shape of each tensor
                should corresponds to `num_centroids`. For example, if `num_centroids` is
                `(K1, K2, K3)`, then `centroids` should be a list of 3 tensors, with shape
                (K1, D), (K2, D), and (K3, D), respectively.
                Returned tensors will be on cpu.
        """

        if predict_only and not isinstance(centroids, list):
            raise ValueError(
                "You specified `predict_only` mode, in which case centroids should be "
                "provided as a list of tensor."
            )

        # initialize local index and data
        if self.local_data is None or self.local_index is None:
            local_data, local_index = get_reaction_features(
                self.model, self.data_loader, self.device
            )
        else:
            local_data = self.local_data.to(self.device)
            local_index = self.local_index.to(self.device)

        # initialize centroids
        if isinstance(centroids, list):
            init_centroids = centroids
        elif centroids == "random" or centroids is None:
            init_centroids = distributed_initialize_centroids(
                local_data, self.num_centroids, self.device
            )
        else:
            raise ValueError(f"Unsupported centroids init methods: {centroids}")

        if predict_only:
            assignments, centroids = distributed_kmeans_predict(
                local_data,
                local_index,
                len(self.data_loader.dataset),
                self.num_centroids,
                init_centroids,
                similarity,
                self.device,
            )
        else:
            assignments, centroids = distributed_kmeans(
                local_data,
                local_index,
                len(self.data_loader.dataset),
                self.num_centroids,
                init_centroids,
                num_iters,
                similarity,
                self.device,
            )

        return assignments, centroids

    def set_local_data_and_index(self, data: torch.Tensor, index: torch.Tensor):
        """
        Args:
            data: data distributed to the local process.
                shape (N_local, D), where `N_local` is number of data points distributed to
                the local process and the `D` is the feature dimension. If `None`, will
                automatically get the data from the data loader.
            index: index of the data distributed to the local process. The index is
                the global index of the data point in the data array stored in the dataset.
                For example, index [3,2,5] means the local data are items 3, 2, 5 in the
                dataset.
                shape (N_local,), where `N_local` is number of data points distributed to
                the local process. If `None`, will automatically get the data from the data
                loader.
        """
        self.local_data = data
        self.local_index = index


class ReactionCluster:
    """
    Cluster data using using k-means, with multiple cluster heads.


    Based on `kmeans_pytorch` at: https://github.com/subhadarship/kmeans_pytorch
    """

    def __init__(
        self,
        model,
        data_loader,
        num_centroids: List[int] = (10, 10),
        device=None,
    ):
        super(ReactionCluster, self).__init__()
        self.model = model
        self.data_loader = data_loader
        self.num_centroids = num_centroids
        self.device = device

        self.data = None

    def get_cluster_assignments(
        self,
        num_iters: int = 10,
        similarity: str = "cosine",
        centroids: Union[List[torch.Tensor], str, None] = "random",
        predict_only: bool = False,
        tol=1.0,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:

        if predict_only and not isinstance(centroids, list):
            raise ValueError(
                "You specified `predict_only` mode, in which case centroids should be "
                "provided as a list of tensor."
            )

        if self.data is None:
            data, _ = get_reaction_features(self.model, self.data_loader, self.device)
        else:
            data = self.data

        # initialize centroids
        if isinstance(centroids, list):
            init_centroids = centroids
        elif centroids == "random" or centroids is None:
            init_centroids = [initialize_centroids(data, K) for K in self.num_centroids]

        else:
            raise ValueError(f"Unsupported centroids init methods: {centroids}")

        # apply k-means
        all_assignments = []
        all_centroids = []
        for K, c in zip(self.num_centroids, init_centroids):
            if predict_only:

                if similarity == "euclidean":
                    pairwise_distance_function = pairwise_distance
                elif similarity == "cosine":
                    pairwise_distance_function = pairwise_cosine
                else:
                    raise NotImplementedError
                dis = pairwise_distance_function(data, c, device=self.device)
                assignment = torch.argmin(dis, dim=1)
                ctrd = c

            else:
                assignment, ctrd = kmeans(
                    X=data,
                    num_clusters=K,
                    distance=similarity,
                    cluster_centers=c,
                    tol=tol,
                    tqdm_flag=False,
                    iter_limit=num_iters,
                    device=self.device,
                )
            all_assignments.append(assignment)
            all_centroids.append(ctrd)

        return all_assignments, all_centroids

    def set_local_data_and_index(self, data, index):
        """
        Index is ignored, since this is for running in serial mode.
        """
        self.data = data


def distributed_initialize_centroids(
    local_data: torch.Tensor, num_prototypes: List[int], device=None
) -> List[torch.Tensor]:
    """
    Initialize the centroids from the features of of rank 0 for all cluster heads.
    The number of cluster heads equal the size of `num_centroids`.

    Args:
        local_data: data on the local process.
            shape (N, D), where N is the local number of data points, and D is the
            dimension of the data.
        num_prototypes: number of clusters for each cluster head.
        device:

    Returns:
        all_centroids: each elements is a 2D tensor of shape (K, D), where K is the
            number of centroids for the cluster head, and D is the feature size.
            Returned tensors will be on cpu.
    """
    local_data = local_data.to(device)

    feature_dim = local_data.size(1)

    all_centroids = []

    with torch.no_grad():
        for i_K, K in enumerate(num_prototypes):

            # init centroids
            centroids = torch.empty(K, feature_dim, device=device)

            # broadcast centroids from rank 0
            if dist.get_rank() == 0:
                random_idx = torch.randperm(len(local_data))[:K]
                assert len(random_idx) >= K, (
                    f"Number of data points ({len(local_data)}) smaller than number of "
                    f"centroids ({K}). You may want to add more data."
                )
                centroids = local_data[random_idx]
            dist.broadcast(centroids, 0)

            all_centroids.append(centroids.cpu())

    return all_centroids


def distributed_kmeans(
    local_data: torch.Tensor,
    local_index: torch.Tensor,
    dataset_size: int,
    num_prototypes: List[int],
    centroids: Optional[List[torch.Tensor]] = None,
    num_iters: int = 10,
    similarity: str = "cosine",
    device=None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Cluster the reactions using k-means.

    Args:
        local_data: shape (N_local, D). The local data to cluster. N_local is the
            number of local data points, and D is the feature dimension.
        local_index: shape (N_local,). Index of the local data in the global dataset
            array.
        dataset_size: total size of the dataset (not the local size)
        num_prototypes: number of clusters for each cluster head. The number of
            cluster heads is `len(num_centroids)`. e.g. (K1, K2, K3).
        centroids: initial centroids for the k-means method. If `None`, randomly
            initialize_centroids from data in rank 0 process. Otherwise, should be a list
            of tensor, each giving the centroids for a cluster heads.
            For example, the shapes of tensors is [(K1,D), (K2, D), (K3,D)], where K1,
            K2, K3 are the number of centroids for each cluster heads as given in
            `num_centroids`, and D is the feature dimension.
        num_iters: number of iterations
        similarity: [`cosine`|`l2`] similarity measure for the distance between
            data and centroid.
        device:

    Returns:
        assignments: a list of tensor of shape (N_global,), where N is the total number
            of data points in the dataset. Each tensor is the assignments of the data
            points to clusters in a cluster head. Returned tensors will be on cpu.
        centroids: a list of tensor, each of shape (K, D), where K is the number of
            clusters and D is the feature dimension. Returned tensors will be on cpu.
    """
    local_data = local_data.to(device)
    local_index = local_index.to(device)

    if centroids is None:
        init_centroids = distributed_initialize_centroids(
            local_data, num_prototypes, device
        )
    else:
        init_centroids = centroids

    feature_dim = local_data.size(1)

    if similarity == "cosine":
        # normalize data
        local_data = F.normalize(local_data, dim=1, p=2)

    assignments = [-100 * torch.ones(dataset_size).long() for _ in num_prototypes]
    all_centroids = []

    with torch.no_grad():

        for i_K, K in enumerate(num_prototypes):
            # run distributed k-means

            centroids = init_centroids[i_K].to(device)

            for n_iter in range(num_iters + 1):

                # E step
                if similarity == "cosine":
                    centroids = F.normalize(centroids, dim=1, p=2)
                    distance = torch.mm(local_data, centroids.t())
                    _, local_assignments = distance.max(dim=1)
                elif similarity == "euclidean":
                    # N*1*D
                    A = local_data.unsqueeze(dim=1)
                    # 1*K*D
                    B = centroids.unsqueeze(dim=0)
                    distance = torch.square(A - B).sum(dim=-1)
                    _, local_assignments = distance.min(dim=1)
                else:
                    raise ValueError(f"Unsupported similarity: {similarity}")

                # finish
                if n_iter == num_iters:
                    break

                # M step
                where_helper = get_indices_sparse(local_assignments.cpu().numpy())
                counts = torch.zeros(K).to(device, non_blocking=True).int()
                emb_sums = torch.zeros(K, feature_dim).to(device, non_blocking=True)
                for k in range(len(where_helper)):
                    if len(where_helper[k][0]) > 0:
                        emb_sums[k] = torch.sum(local_data[where_helper[k][0]], dim=0)
                        counts[k] = len(where_helper[k][0])
                dist.all_reduce(counts)
                mask = counts > 0
                dist.all_reduce(emb_sums)
                centroids[mask] = emb_sums[mask] / counts[mask].unsqueeze(1)

            # assign centroids back
            all_centroids.append(centroids.cpu())

            # gather the assignments
            assignments_all = torch.empty(
                dist.get_world_size(),
                local_assignments.size(0),
                dtype=local_assignments.dtype,
                device=local_assignments.device,
            )
            assignments_all = list(assignments_all.unbind(0))
            dist_process = dist.all_gather(
                assignments_all, local_assignments, async_op=True
            )
            dist_process.wait()
            assignments_all = torch.cat(assignments_all).cpu()

            # gather the indexes
            indexes_all = torch.empty(
                dist.get_world_size(),
                local_index.size(0),
                dtype=local_index.dtype,
                device=local_index.device,
            )
            indexes_all = list(indexes_all.unbind(0))
            dist_process = dist.all_gather(indexes_all, local_index, async_op=True)
            dist_process.wait()
            indexes_all = torch.cat(indexes_all).cpu()

            # assign assignments
            assignments[i_K][indexes_all] = assignments_all

        centroids = all_centroids

    return assignments, centroids


def distributed_kmeans_predict(
    local_data: torch.Tensor,
    local_index: torch.Tensor,
    dataset_size: int,
    num_prototypes: List[int],
    init_centroids: Optional[List[torch.Tensor]],
    similarity: str = "cosine",
    device=None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Given cluster centroids, predict the assignments.

    Similar to distributed_kmeans(), but do not do the centroids update step.
    To be more specific, this is the same as distributed_kmeans() with num_iters = 0
    and without the M step.
    """
    local_data = local_data.to(device)
    local_index = local_index.to(device)

    if similarity == "cosine":
        # normalize data
        local_data = F.normalize(local_data, dim=1, p=2)

    assignments = [-100 * torch.ones(dataset_size).long() for _ in num_prototypes]

    with torch.no_grad():

        for i_K, K in enumerate(num_prototypes):

            centroids = init_centroids[i_K].to(device)

            # E step
            if similarity == "cosine":
                centroids = F.normalize(centroids, dim=1, p=2)
                distance = torch.mm(local_data, centroids.t())
                _, local_assignments = distance.max(dim=1)
            elif similarity == "euclidean":
                # N*1*D
                A = local_data.unsqueeze(dim=1)
                # 1*K*D
                B = centroids.unsqueeze(dim=0)
                distance = torch.square(A - B).sum(dim=-1)
                _, local_assignments = distance.min(dim=1)
            else:
                raise ValueError(f"Unsupported similarity: {similarity}")

            # gather the assignments
            assignments_all = torch.empty(
                dist.get_world_size(),
                local_assignments.size(0),
                dtype=local_assignments.dtype,
                device=local_assignments.device,
            )
            assignments_all = list(assignments_all.unbind(0))
            dist_process = dist.all_gather(
                assignments_all, local_assignments, async_op=True
            )
            dist_process.wait()
            assignments_all = torch.cat(assignments_all).cpu()

            # gather the indexes
            indexes_all = torch.empty(
                dist.get_world_size(),
                local_index.size(0),
                dtype=local_index.dtype,
                device=local_index.device,
            )
            indexes_all = list(indexes_all.unbind(0))
            dist_process = dist.all_gather(indexes_all, local_index, async_op=True)
            dist_process.wait()
            indexes_all = torch.cat(indexes_all).cpu()

            # assign assignments
            assignments[i_K][indexes_all] = assignments_all

    return assignments, init_centroids


def get_indices_sparse(data):
    cols = np.arange(data.size)
    M = csr_matrix((cols, (data.ravel(), cols)), shape=(int(data.max()) + 1, data.size))
    return [np.unravel_index(row.data, data.shape) for row in M]


def initialize_centroids(X, num_clusters):
    """
    initialize_centroids cluster centers
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :return: (np.array) initial state
    """
    num_samples = len(X)
    indices = np.random.choice(num_samples, num_clusters, replace=False)
    initial_state = X[indices]
    return initial_state


def kmeans(
    X,
    num_clusters,
    distance="euclidean",
    cluster_centers=None,
    tol=1e-4,
    tqdm_flag=True,
    iter_limit=0,
    device=torch.device("cpu"),
):
    """
    perform kmeans
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param tol: (float) threshold [default: 0.0001]
    :param device: (torch.device) device [default: cpu]
    :param tqdm_flag: Allows to turn logs on and off
    :param iter_limit: hard limit for max number of iterations
    :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
    """

    if distance == "euclidean":
        pairwise_distance_function = pairwise_distance
    elif distance == "cosine":
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError

    # transfer to device
    X = X.to(device)

    # initialize_centroids
    if cluster_centers is None:
        initial_state = initialize_centroids(X, num_clusters)
    else:
        # find data point closest to the initial cluster center
        initial_state = cluster_centers
        dis = pairwise_distance_function(X, initial_state, device=device)
        choice_points = torch.argmin(dis, dim=0)
        initial_state = X[choice_points]
        initial_state = initial_state.to(device)

    iteration = 0
    if tqdm_flag:
        tqdm_meter = tqdm(desc="[running k-means]")
    while True:

        dis = pairwise_distance_function(X, initial_state, device=device)

        choice_cluster = torch.argmin(dis, dim=1)

        initial_state_pre = initial_state.clone()

        for index in range(num_clusters):
            selected = (
                torch.nonzero(choice_cluster == index, as_tuple=False)
                .squeeze()
                .to(device)
            )

            selected = torch.index_select(X, 0, selected)

            # only update nonempty clusters
            if len(selected) != 0:
                initial_state[index] = selected.mean(dim=0)

        center_shift = torch.sum(
            torch.sqrt(torch.sum((initial_state - initial_state_pre) ** 2, dim=1))
        )

        # increment iteration
        iteration = iteration + 1

        # update tqdm meter
        if tqdm_flag:
            tqdm_meter.set_postfix(
                iteration=f"{iteration}",
                center_shift=f"{center_shift ** 2:0.6f}",
                tol=f"{tol:0.6f}",
            )
            tqdm_meter.update()
        if center_shift ** 2 < tol:
            break
        if iter_limit != 0 and iteration >= iter_limit:
            break

    return choice_cluster.cpu(), initial_state.cpu()


def pairwise_distance(data1, data2, device=None):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    dis = (A - B) ** 2.0
    # return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1).squeeze()

    return dis


def pairwise_cosine(data1, data2, device=None):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)

    cosine = A_normalized * B_normalized

    # return N*N matrix for pairwise distance
    cosine_dis = 1 - cosine.sum(dim=-1).squeeze()

    return cosine_dis


def get_reaction_features(
    model, data_loader, device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get all the reaction features and the indices of samples in the dataset.

    Args:
        model:
        data_loader:
        device:

    Returns:
        indices: the indices of samples in the dataset (the global index, not the local
            index).
            shape (N,): where N is the total number of samples obtainable from the
            data_loader (not the total number of samples in the dataset if using
            DistributedSampler).
            Returned tensor will be on device.
        feats: the reaction features.
            shape (N, D): where N is the total number of samples obtainable from the
            data_loader (not the total number of samples in the dataset if
            using DistributedSampler), and D is the feature dimension.
            Returned tensor will be on device.

    """

    nodes = ["atom", "bond", "global"]

    all_indices = []
    all_feats = []
    with torch.no_grad():
        for batch in data_loader:
            batch = move_data_to_device(batch, device)
            indices, mol_graphs, rxn_graphs, labels, metadata = batch

            mol_graphs = mol_graphs.to(device)
            rxn_graphs = rxn_graphs.to(device)
            feats = {nt: mol_graphs.nodes[nt].data.pop("feat") for nt in nodes}

            feats, rxn_feats = model(mol_graphs, rxn_graphs, feats, metadata)
            preds = model.decode(feats, rxn_feats, metadata)

            all_indices.append(indices.to(device))
            all_feats.append(preds["reaction_cluster"].detach())

    indices = torch.cat(all_indices)
    feats = torch.cat(all_feats)

    return feats, indices
