"""
A k-mean clustering method running only on a single GPU.

The kmeans function is based on `kmeans_pytorch` at: https://github.com/subhadarship/kmeans_pytorch
"""

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from rxnrep.data.uspto import collate_fn


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
            feats = get_reaction_features(self.model, self.data_loader, self.device)

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


def pairwise_distance(data1, data2, device=torch.device("cpu")):
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


def pairwise_cosine(data1, data2, device=torch.device("cpu")):
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


def get_reaction_features(model, data_loader, device) -> torch.Tensor:
    """
    Get all the reaction features.

    Args:
        model:
        data_loader:
        device:

    Returns:
        2D tensor of shape (N, D), where N is the total number of reactions in the
        dataset, and D is the dimension of the reaction feature.

    """
    model.eval()

    nodes = ["atom", "bond", "global"]

    all_feats = []
    with torch.no_grad():
        for it, (mol_graphs, rxn_graphs, labels, metadata) in enumerate(data_loader):
            mol_graphs = mol_graphs.to(device)
            rxn_graphs = rxn_graphs.to(device)
            feats = {nt: mol_graphs.nodes[nt].data.pop("feat").to(device) for nt in nodes}

            fts = model.get_reaction_features(mol_graphs, rxn_graphs, feats, metadata)
            all_feats.append(fts.detach())

    all_feats = torch.cat(all_feats)

    return all_feats
