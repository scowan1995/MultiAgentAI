from .basic_bidding_agent import BasicBiddingAgent

import numpy.random
import statistics


class RandomBiddingAgent(BasicBiddingAgent):
    def __init__(self, training_set, initial_budget):
        self._mean_val = 0
        self._std_val = 0
        self._lower_bound = 0
        self._upper_bound = 0
        super().__init__(training_set, initial_budget)

    def _train(self, training_set):
        """
        Training in this case will invole getting the mean and standard deviation of the
        distribution of the training set to create a gaussian to sample from.
        """
        success_bid_dist = training_set.data_targets.loc[lambda x: x["click"] == 1, "bidprice"]
        self._mean_val = success_bid_dist.mean()
        self._std_val = success_bid_dist.std()
        print("Mean value", self._mean_val)
        print("Standard dev", self._std_val)

        self._lower_bound = self._mean_val - (3 * self._std_val)
        self._upper_bound = self._mean_val + (3 * self._std_val)
        print("lower bound", self._lower_bound)
        print("upper bound", self._upper_bound)

    def _bidding_function(self, utility=None, cost=None):
        """Deploy the learned bidding model"""
        bid = numpy.random.uniform(self._lower_bound, self._upper_bound)
        # print("bid chosen", bid)
        return bid

    def set_random_bidding_bounds(self, lower_bound=0.0, upper_bound=1.0):
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

    def perturb_boundaries(self):
        self._lower_bound = numpy.random.uniform(self._lower_bound, self._mean_val)
        self._upper_bound = numpy.random.uniform(self._mean_val, self._upper_bound)


"""
    def get_test_bounds(self, nbounds):
        bounds = []
        for i in range(nbounds):
            low = numpy.random.uniform(self._lower_bound, self._mean_val)
            high = numpy.random.uniform(self._mean_val, self._upper_bound)
            bounds.append((low, high))
        return bounds

    def tune(self, ntests=100):
        bounds = self.get_test_bounds(ntests)
        print("Bounds used:")
        cpcs = []
        ctrs = []
        print(bounds)
        best_cpc = 1000000
        ctr = 0
        best_bound = 0.0, 1.0
        for ind, bound in enumerate(bounds):
            print("On test", ind)
            new_cpc, new_ctr = self.begin_bidding(bound[0], bound[1])
            cpcs.append(new_cpc)
            ctrs.append(new_ctr)
            if new_cpc < best_cpc:
                best_cpc = new_cpc
                ctr = new_ctr
                best_bound = bound
            print("Using bound", bound)
            print("acheived", new_cpc)
            print("With CTR", new_ctr)

        print("Tuning finished with", ntests, " tests in total")
        print("best cpc", best_cpc)
        print("got ctr:", ctr)
        print("using", best_bound)
        print("The mean cpc during tuning was", statistics.mean(cpcs))
        print(
            "the variance and standard deviance respectively were",
            statistics.variance(cpcs),
            " ,",
            statistics.stdev(cpcs),
        )
        print(
            "the variance and standard deviance respectively were",
            statistics.variance(ctrs),
            " ,",
            statistics.stdev(ctrs),
        )


    def begin_bidding(self, low, high):
        bids_won = 0
        total_clicks = 0
        total_paid = 0.0

        budget = self.initial_budget
        for index, row in self.df.iterrows():
            #  print(low)
            #  print(high)
            bid = self._bidding_function(low, high)
            if bid > budget:
                bid = budget
            if bid >= row["bidprice"]:
                bids_won += 1
                #     print("bid won")
                #     print(low, high)
                if row["click"] == 1:
                    if budget >= row["payprice"]:
                        budget -= bid
                        total_clicks += 1
                        total_paid += bid
        print("total clicks", total_clicks)
        print("total paid", total_paid)
        print("bids won", bids_won)
        if total_clicks > 0:
            cpc = total_paid / total_clicks
        else:
            cpc = 0

        if bids_won > 0:
            ctr = total_clicks / bids_won
        else:
            ctr = 0
        print("CTR:", ctr)
        print("CPC", cpc)
        return cpc, ctr
"""
