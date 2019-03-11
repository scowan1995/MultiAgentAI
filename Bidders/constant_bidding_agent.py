from .basic_bidding_agent import BasicBiddingAgent


class ConstantBiddingAgent(BasicBiddingAgent):

    def __init__(self, training_set, initial_budget):
        super().__init__(training_set, initial_budget)

    def _train(self, training_set):
        """TODO: explore data (use class DataExploration) to determine reasonable constant bids"""
        self._bid_value = 277
        pass

    def _bidding_function(self):
        """Deploy the learned bidding model"""
        return self._bid_value

    def set_const_bid_value(self, bid):
        self._bid_value = bid



