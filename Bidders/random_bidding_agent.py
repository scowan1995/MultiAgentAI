from basic_bidding_agent import BasicBiddingAgent
import random
import os
import pandas as pd
import numpy.random as r
import statistics as s


class RandomBiddingAgent(BasicBiddingAgent):
    def __init__(self, training_set, initial_budget):
        super().__init__(training_set, initial_budget)
        self.initial_budget = initial_budget
        self.mean_val = 0
        self.std_val = 0
        self.lower_bound = 0
        self.upper_bound = 0

    def _train(self, training_set):
        """
        Training in this case will invole getting the mean and standard deviation of the
        distribution of the training set to create a gaussian to sample from.
        """
        data_path = os.path.abspath(os.pardir + "/MultiAgentAI/Data/train.csv")
        self.df = pd.read_csv(data_path, na_values=["Na", "null"]).fillna(0)
        success_bid_dist = self.df.loc[lambda x: x["click"] == 1, "bidprice"]
        self.mean_val = success_bid_dist.mean()
        self.std_val = success_bid_dist.std()
        print("Mean value", self.mean_val)
        print("Standard dev", self.std_val)
        self.lower_bound = self.mean_val - (3 * self.std_val)
        self.upper_bound = self.mean_val + (3 * self.std_val)
        print("lower bound", self.lower_bound)
        print("upper bound", self.upper_bound)

    def _bidding_function(self, low=0.0, high=1.0):
        """Deploy the learned bidding model"""
        bid = r.uniform(low, high)
        # print("bid chosen", bid)
        return bid

    def get_test_bounds(self, nbounds):
        bounds = []
        for i in range(nbounds):
            low = r.uniform(self.lower_bound, self.mean_val)
            high = r.uniform(self.mean_val, self.upper_bound)
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
        print("The mean cpc during tuning was", s.mean(cpcs))
        print(
            "the variance and standard deviance respectively were",
            s.variance(cpcs),
            " ,",
            s.stdev(cpcs),
        )
        print(
            "the variance and standard deviance respectively were",
            s.variance(ctrs),
            " ,",
            s.stdev(ctrs),
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
