from .basic_bidding_agent import BasicBiddingAgent

import numpy.random
import statistics


class RandomBiddingAgent(BasicBiddingAgent):
    def __init__(self, training_set, initial_budget):
        self._mean_val = 0
        self._std_val = 0
        self._lower_bound = 0
        self._upper_bound = 0

        # If perturbation is true the agent bids with boundaries perturbed with respect to the learned ones
        self.perturbation = False
        super().__init__(training_set, initial_budget)

    def _train(self, training_set):
        """
        Training in this case will involve getting the mean and standard deviation of the
        distribution of the training set to create a gaussian to sample from.
        """
        success_bid_dist = training_set.data_targets.loc[lambda x: x["click"] == 1, "payprice"]
        self._mean_val = success_bid_dist.mean()
        self._std_val = success_bid_dist.std()

        self._lower_bound = max(0, self._mean_val - (3 * self._std_val))
        self._upper_bound = max(0, self._mean_val + (3 * self._std_val))

    def _bidding_function(self, utility=None, cost=None):
        """Deploy the learned bidding model"""
        if self.perturbation:
            low, high = self._perturb_boundaries()
            bid = numpy.random.uniform(low, high)
        else:
            bid = numpy.random.uniform(self._lower_bound, self._upper_bound)
        # print("bid chosen", bid)
        return bid

    def _perturb_boundaries(self):
        lower = numpy.random.uniform(self._lower_bound - self._std_val, self._mean_val)
        upper = numpy.random.uniform(self._mean_val, self._upper_bound + self._std_val)
        return lower, upper

    def set_boundaries(self, lower_bound, upper_bound):
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

    def get_boundaries(self):
        return self._lower_bound, self._upper_bound

    def get_mean(self):
        return self._mean_val
