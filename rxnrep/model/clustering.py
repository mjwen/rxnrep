"""
A k-mean cluster method running only on a single GPU.

The kmeans function is based on `kmeans_pytorch` at: https://github.com/subhadarship/kmeans_pytorch
"""

import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from rxnrep.data.uspto import collate_fn
from typing import Tuple, List, Optional


class DistributedReactionCluster:
    """
    Distributed K-mean clustering.

    Adapted from deepcluster-v2 at:
    https://github.com/facebookresearch/swav/blob/master/main_deepclusterv2.py

    Args:

        dataset_size: total dataset size (all samples in all processes)
        num_prototypes: each element gives the number of clusters in a cluster head.
            A total number of `len(num_prototypes)` cluster heads are used.
    """

    def __init__(
        self,
        model,
        data_loader,
        batch_size: int,
        dataset_size: int,
        num_prototypes=(1000, 1000),
        local_data=None,
        local_index=None,
        centroids=None,
        device=None,
    ):
        super(DistributedReactionCluster, self).__init__()
        self.model = model
        self.data_loader = data_loader
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.num_prototypes = num_prototypes

        self.local_data = local_data
        self.local_index = local_index
        self.centroids = centroids

        self.device = device

    def get_cluster_assignments(
        self, num_iters: int = 10, centroids_init="random", similarity: str = "cosine"
    ):
        """
        Get the assignments of the data points to the clusters.

        Note, there are multiple cluster heads, determined by `num_prototypes`.



        Args:
            centroids_init: [`random`|`last`] methods to initialize centroids.
                If `random`, will randomly select centroids from the embedding data.
                If `last`, will use the centroids the last time clustering is run.
            similarity: [`cosine`|`l2`] similarity measure for the distance between
                data and centroid.

        Returns:

        """
        # initialize local index and data
        if self.local_data is None or self.local_index is None:
            self.local_data, self.local_index = get_reaction_features(
                self.model, self.data_loader, self.device
            )
        local_index = self.local_memory_index
        local_data = self.local_memory_embeddings

        # initialize centroids
        if centroids_init == "random":
            centroids = distributed_initialize_centroids(
                local_data, self.num_prototypes, self.device
            )
        elif centroids_init == "last":
            centroids = self.centroids
        else:
            raise ValueError(f"Unsupported centroids init methods: {centroids_init}")

        assignments, centroids = distributed_kmeans(
            local_data,
            local_index,
            self.dataset_size,
            self.num_prototypes,
            centroids,
            num_iters,
            similarity,
            self.device,
        )

        if centroids_init == "last":
            self.centroids = centroids

        return assignments

    def set_local_memory_index_and_embeddings(
        self, index: torch.Tensor, embeddings: torch.Tensor
    ):
        """
        Args:
            index: shape (N,): index of the memory embeddings
            embeddings:  shape (N,D): memory embeddings
        """
        self.local_memory_index = index
        self.local_memory_embeddings = embeddings


class ReactionCluster:
    """
    Cluster the reaction features using k-means.
    """

    def __init__(
        self,
        model,
        dataset,
        num_clusters=10,
        iter_limit=500,
        batch_size=1000,
        device=None,
    ):
        super(ReactionCluster, self).__init__()
        self.model = model
        self.num_clusters = num_clusters
        self.iter_limit = iter_limit
        self.device = device
        self.data_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )

        self.cluster_ids = None
        self.cluster_centers = None
        self.epoch = None

    def get_cluster_ids(self, epoch: int):
        generate = self.whether_to_generate_ids(epoch)

        if generate:
            print(f"Generating new cluster ids at epoch {epoch}...")
            # get features
            feats, _ = get_reaction_features(self.model, self.data_loader, self.device)

            # apply k-means
            cluster_ids, cluster_centers = kmeans(
                X=feats,
                num_clusters=self.num_clusters,
                distance="euclidean",
                cluster_centers=None,
                tol=1.0,
                tqdm_flag=True,
                iter_limit=self.iter_limit,
                device=self.device,
            )
            self.cluster_ids = cluster_ids
            self.cluster_centers = cluster_centers

        return self.cluster_ids

    def whether_to_generate_ids(self, epoch: int) -> bool:
        """
        Determine whether to generate new cluster labels.

        Now, we hard-code it te generate every 4 epochs.

        Args:
            epoch: current epoch
        """
        if self.epoch is None or epoch % 4 == 0:
            generate = True
        else:
            generate = False

        self.epoch = epoch

        return generate


def distributed_initialize_centroids(
    local_data: torch.Tensor, num_prototypes: Tuple[int], device=None
) -> List[torch.Tensor]:
    """
    Initialize the centroids from the features of of rank 0 for all cluster heads.
    The number of cluster heads equal the size of `num_prototypes`.

    Args:
        local_data: data on the local process.
            shape (N, D), where N is the local number of data points, and D is the
            dimension of the data.
        num_prototypes: number of clusters for each cluster head.

    Returns:
        centroids: each elements is a 2D tensor of shape (K, D), where K is the
            number of centroids for the cluster head, and D is the feature size.
    """

    feature_dim = local_data.size(1)

    all_centroids = []
    with torch.no_grad():
        for i_K, K in enumerate(num_prototypes):

            # init centroids with elements from memory bank of rank 0
            centroids = torch.empty(K, feature_dim).to(device, non_blocking=True)
            if dist.get_rank() == 0:
                random_idx = torch.randperm(len(local_data))[:K]
                assert len(random_idx) >= K, "please reduce the number of centroids"
                centroids = local_data[random_idx]
            dist.broadcast(centroids, 0)

            all_centroids.append(centroids.detach().cpu())

    return all_centroids


def distributed_kmeans(
    local_data: torch.Tensor,
    local_index: torch.Tensor,
    dataset_size: int,
    num_prototypes: Tuple[int],
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
            cluster heads is `len(num_prototypes)`. e.g. (K1, K2, K3).
        centroids: initial centroids for the k-means method. If `None`, randomly
            initialize from data in rank 0 process. Otherwise, should be a list of
            tensor, each giving the centroids for a cluster heads.
            For example, the shapes of tensors is [(K1,D), (K2, D), (K3,D)], where K1,
            K2, K3 are the number of centroids for each cluster heads as given in
            `num_prototypes`, and D is the feature dimension.
        num_iters: number of iterations
        similarity: [`cosine`|`l2`] similarity measure for the distance between
            data and centroid.

    Returns:
        assignments: a list of tensor of shape (N_global,), where N is the total number
            of data points in the dataset. Each tensor is the assignments of the data
            points to clusters in a cluster head.
        centroids: a list of tensor, each of shape (K, D), where K is the number of
            clusters and D is the feature dimension.
    """

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

            # initialize centroids
            if init_centroids is None:
                centroids = distributed_initialize_centroids(
                    local_data, num_prototypes, device
                )
            # use passed in centroids
            else:
                centroids = init_centroids[i_K]
            centroids = centroids.to(device)

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
            all_centroids[i_K] = centroids.cpu()

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


def get_indices_sparse(data):
    cols = np.arange(data.size)
    M = csr_matrix((cols, (data.ravel(), cols)), shape=(int(data.max()) + 1, data.size))
    return [np.unravel_index(row.data, data.shape) for row in M]


def initialize(X, num_clusters):
    """
    initialize cluster centers
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

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    # initialize
    if cluster_centers is None:
        initial_state = initialize(X, num_clusters)
    else:
        print("resuming")
        # find data point closest to the initial cluster center
        initial_state = cluster_centers
        dis = pairwise_distance_function(X, initial_state)
        choice_points = torch.argmin(dis, dim=0)
        initial_state = X[choice_points]
        initial_state = initial_state.to(device)

    iteration = 0
    if tqdm_flag:
        tqdm_meter = tqdm(desc="[running k-means]")
    while True:

        dis = pairwise_distance_function(X, initial_state)

        choice_cluster = torch.argmin(dis, dim=1)

        initial_state_pre = initial_state.clone()

        for index in range(num_clusters):
            selected = (
                torch.nonzero(choice_cluster == index, as_tuple=False)
                .squeeze()
                .to(device)
            )

            selected = torch.index_select(X, 0, selected)

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

            where N is the total number of samples in
            this process (data_loader).
        feats: the reaction features.
            shape (N, D): where N is the total number of samples obtainable from the
            data_loader (not the total number of samples in the dataset if
            using DistributedSampler), and D is the feature dimension.

    """

    nodes = ["atom", "bond", "global"]

    all_indices = []
    all_feats = []
    with torch.no_grad():
        for it, (indices, mol_graphs, rxn_graphs, labels, metadata) in enumerate(
            data_loader
        ):
            all_indices.append(indices.to(device))

            mol_graphs = mol_graphs.to(device)
            rxn_graphs = rxn_graphs.to(device)
            feats = {nt: mol_graphs.nodes[nt].data.pop("feat").to(device) for nt in nodes}

            _, fts = model(mol_graphs, rxn_graphs, feats, metadata)
            all_feats.append(fts.detach())

    indices = torch.cat(all_indices)
    feats = torch.cat(all_feats)

    return feats, indices
