import random
from .basic_bidding_agent import BasicBiddingAgent


class ConstantBiddingAgent(BasicBiddingAgent):
    def __init__(self, training_set, initial_budget):
        super().__init__(training_set, initial_budget)
        self.lower_value_range = 0
        self.higher_value_range = 0

    def _train(self, training_set):
        """TODO: explore data (use class DataExploration) to determine a reasonable range"""
        # self.lower_value_range = ...
        # self.higher_value_range = ...
        pass

    def _bidding_function(self, utility=None, cost=None):
        """Deploy the learned bidding model"""
        return random.uniform(self.lower_value_range, self.higher_value_range)
