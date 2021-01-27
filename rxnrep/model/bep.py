from typing import List, Tuple

import torch


class ActivationEnergyPredictor:
    """
    Predict activation energy from reaction energy based on BEP.

    Note, not every reaction will have a predicted activation energy. This is
    determined by `min_num_data_points_for_fitting` and see below.

    Args:
        num_centroids: number of clusters in each clustering prototype.
        min_num_data_points_for_fitting: minimum number of data points used to fit
                the BEP for a cluster. At least 2 data points are needed to fit the
                linear regression model. If some cluster has fewer reactions with
                true activation energies, we cannot fit a BEP and thus cannot predict
                the activation energies for other reactions in the cluster. In this
                case, their corresponding value are set to 0.0, and they can be
                identified in ``have_predicted_activation_energy`` returned by
                ``predict`` by setting the value to `False`.
        device:
    """

    def __init__(
        self,
        num_centroids: List[int],
        min_num_data_points_for_fitting: int = 2,
        device="cpu",
    ):
        self.num_centroids = num_centroids
        self.min_num_data_points_for_fitting = min_num_data_points_for_fitting
        self.device = device

        self.regressors = {}

    def fit(
        self,
        reaction_energy: torch.Tensor,
        activation_energy: torch.Tensor,
        have_activation_energy: torch.Tensor,
        assignments: List[torch.Tensor],
        predict: bool = False,
    ):
        """
        Args:
            reaction_energy: 1D tensor, reaction energy of all reactions in the dataset.
            activation_energy: 1D tensor, activation energy of all reactions in the
                dataset.
            have_activation_energy: 1D tensor, whether a reaction have activation energy;
                If `False`, the corresponding element in activation_energy should be
                regarded as None (i.e. not exist).
            assignments: list of 1D tensor of of shape (N,), where N is the number of
                reactions in the dataset. Cluster assignment of data points (
                reactions), each tensor corresponds to one clustering prototype.
            predict: whether to make predictions. Return tensors will be on self.device.
        """
        assert len(assignments) == len(
            self.num_centroids
        ), "assignments size != len(num_centroids)"

        reaction_energy = reaction_energy.to(self.device)
        activation_energy = activation_energy.to(self.device)
        have_activation_energy = have_activation_energy.to(self.device)

        with torch.no_grad():

            if predict:
                N = len(reaction_energy)
                predicted_activation_energy = [
                    torch.zeros(N, device=self.device) for _ in self.num_centroids
                ]
                have_predicted_activation_energy = [
                    torch.zeros(N, dtype=torch.bool, device=self.device)
                    for _ in self.num_centroids
                ]

            for i_K, K in enumerate(self.num_centroids):

                assignment = assignments[i_K].to(self.device)

                for i in range(K):

                    # select data points in cluster i
                    indices = assignment == i
                    rxn_energies = reaction_energy[indices]
                    act_energies = activation_energy[indices]
                    have_act_energy = have_activation_energy[indices]

                    # select data points having activation energies
                    rxn_e = rxn_energies[have_act_energy]
                    act_e = act_energies[have_act_energy]

                    # fit a linear regression model
                    if len(act_e) >= self.min_num_data_points_for_fitting:
                        # fit BEP model using data points having activation energies
                        reg = LinearRegression()
                        reg.fit(rxn_e, act_e)
                        self.regressors[(i_K, i)] = reg

                        if predict:
                            # predict activation energy for all data points in cluster
                            pred_act_energies = reg.predict(rxn_energies)
                            predicted_activation_energy[i_K][
                                indices
                            ] = pred_act_energies
                            have_predicted_activation_energy[i_K][indices] = True

                    else:
                        self.regressors[(i_K, i)] = None

        if predict:
            return predicted_activation_energy, have_predicted_activation_energy
        else:
            return None

    def fit_predict(
        self,
        reaction_energy: torch.Tensor,
        activation_energy: torch.Tensor,
        have_activation_energy: torch.Tensor,
        assignments: List[torch.Tensor],
    ):
        """
        Fit and predict.
        """
        return self.fit(
            reaction_energy,
            activation_energy,
            have_activation_energy,
            assignments,
            predict=True,
        )

    def predict(
        self,
        reaction_energy: torch.Tensor,
        assignments: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Predict BEP activation energy.

        Returned tensors will be on self.device.

        Returns:
            predicted_activation_energy: 1D tensor of shape (N,), where N is the number
                of reactions in the dataset. The value for the elements does not
                ``have_predicted_activation_energy`` are set to 0, but they should not
                be used. They can be selected by ``have_predicted_activation_energy``
                Returned tensor on self.device.
            have_predicted_activation_energy: 1D tensor of shape (N,). Whether predicted
                activation energy exists. Returned tensor on self.device.
        """
        reaction_energy = reaction_energy.to(self.device)

        with torch.no_grad():

            N = len(reaction_energy)
            predicted_activation_energy = [
                torch.zeros(N, device=self.device) for _ in self.num_centroids
            ]
            have_predicted_activation_energy = [
                torch.zeros(N, dtype=torch.bool, device=self.device)
                for _ in self.num_centroids
            ]

            for i_K, K in enumerate(self.num_centroids):

                assignment = assignments[i_K].to(self.device)

                for i in range(K):

                    # select data points in cluster i
                    indices = assignment == i

                    if any(indices):
                        rxn_energies = reaction_energy[indices]

                        reg = self.regressors[(i_K, i)]
                        if reg is not None:
                            pred_act_energies = reg.predict(rxn_energies)
                            predicted_activation_energy[i_K][
                                indices
                            ] = pred_act_energies
                            have_predicted_activation_energy[i_K][indices] = True

        return predicted_activation_energy, have_predicted_activation_energy


class LinearRegression:
    """
    Fitting a simple 1D linear model between x and y.

    Equations can be simply derived, e.g. see Intro statistical learning with R, p62.
    """

    def __init__(self):
        self.slope = None
        self.intercept = None

    def fit(self, x: torch.Tensor, y: torch.Tensor):
        """
        Args:
            x: 1D tensor of input
            y: 1D tensor of output

        Returns:
            slope: slope of the linear model
            intercept: of the linear model
        """
        assert len(x) == len(y), f"Expect len(x) == len(y); got {len(x)} and {len(y)}"
        assert len(x) >= 2, f"Expect len(x) >=2; got {len(x)}"

        x_mean = torch.mean(x)
        y_mean = torch.mean(y)
        x_diff = x - x_mean
        y_diff = y - y_mean

        self.slope = torch.sum(torch.dot(x_diff, y_diff)) / torch.sum(
            torch.square(x_diff)
        )
        self.intercept = y_mean - self.slope * x_mean

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.slope * x + self.intercept
