from .basic_bidding_agent import BasicBiddingAgent


class ConstantBiddingAgent(BasicBiddingAgent):

    def __init__(self, training_set, initial_budget):
        self.constant_bid_value = 0
        super().__init__(training_set, initial_budget)


    def _train(self, training_set):
        """TODO: explore data (use class DataExploration) to determine reasonable constant bids"""
        self.constant_bid_value = 10
        pass

    def _bidding_function(self, utility=None, cost=None):
        """Deploy the learned bidding model"""
        return self.constant_bid_value



