from typing import List, Tuple

import torch


class ActivationEnergyPredictor:
    """
    Predict activation energy from reaction energy based on BEP.

    Args:
        reaction_energy: 1D tensor, reaction energy of all reactions in the dataset.
        activation_energy: 1D tensor, activation energy of all reactions in the dataset.
        have_activation_energy: 1D tensor, whether a reaction have activation energy;
            If `False`, the corresponding element in activation_energy should be
            regarded as None (i.e. not exist).
    """

    def __init__(
        self,
        reaction_energy: torch.Tensor,
        activation_energy: torch.Tensor,
        have_activation_energy: torch.Tensor,
    ):
        self.reaction_energy = reaction_energy
        self.activation_energy = activation_energy
        self.have_activation_energy = have_activation_energy

    @classmethod
    def from_data_loader(cls, data_loader):
        """
        We use the get_property() of dataset to get all reaction energy, activation
        energy, have activation energy label. Alternatively, this can be obtained by
        iterating over the data_loader and getting from label and metadata.
        """
        dataset = data_loader.dataset
        reaction_energy = dataset.get_property("reaction_energy")
        activation_energy = dataset.get_property("activation_energy")
        have_activation_energy = dataset.get_property("have_activation_energy")
        return cls(reaction_energy, activation_energy, have_activation_energy)

    def get_predicted_activation_energy_multi_prototype(
        self, assignments: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Predict the activation energy when multiple cluster prototypes are used. This
        is simply calling ``get_predicted_activation_energy()`` multiple times,
        each with a different assignment.
        """
        out = [self.get_predicted_activation_energy(a) for a in assignments]
        predicted_activation_energy, have_predicted_activation_energy = map(
            list, zip(*out)
        )

        return predicted_activation_energy, have_predicted_activation_energy

    def get_predicted_activation_energy(
        self,
        assignment: torch.Tensor,
        minimum_activation_energy_for_bde_fitting: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict the activation energy separately for each cluster.

        Args:
            assignment: 1D tensor of of shape (N,), where N is the number of reactions
                in the dataset. Cluster assignment of data points (reactions).
            minimum_activation_energy_for_bde_fitting: minimum data points used to fit
                the BEP for a cluster. At least 2 data points are needed to fit the
                linear regression model. If some cluster has fewer reactions having
                true activation energies, we cannot fit a BEP and thus cannot predict
                the activation energies for other reactions in the cluster. In this
                case, the corresponding element in the returned
                ``have_predicted_activation_energy`` is set to False.

        Returns:
            predicted_activation_energy: 1D tensor of shape (N,). Returned tensor on CPU.
            have_predicted_activation_energy: 1D tensor of shape (N,). Whether predicted
                activation energy exists. Returned tensor on CPU.
        """

        # move to cpu, since 1) self.reaction_energy ... are on cpu; 2) there are
        # simple computations, no need to move to gpu
        assignment = assignment.to("cpu")

        with torch.no_grad():

            predicted_activation_energy = torch.zeros(assignment.shape)
            have_predicted_activation_energy = torch.zeros(
                assignment.shape, dtype=torch.bool
            )

            max_cluster_index = max(assignment)
            for i in range(max_cluster_index + 1):

                # select data points in cluster i
                indices = assignment == i
                rxn_energies = self.reaction_energy[indices]
                act_energies = self.activation_energy[indices]
                have_act_energy = self.have_activation_energy[indices]

                # select data points having activation energies
                rxn_e = rxn_energies[have_act_energy]
                act_e = act_energies[have_act_energy]

                # fit a linear regression model
                if len(rxn_e) >= minimum_activation_energy_for_bde_fitting:
                    # fit BEP model using data points having activation energies
                    reg = LinearRegression()
                    reg.fit(rxn_e, act_e)

                    # predict activation energy for all data points in cluster
                    pred_act_energies = reg.predict(rxn_energies)
                    predicted_activation_energy[indices] = pred_act_energies
                    have_predicted_activation_energy[indices] = True

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
