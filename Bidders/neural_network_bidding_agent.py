from .basic_bidding_agent import BasicBiddingAgent


class NeuralNetworkBiddingAgent(BasicBiddingAgent):
    def __init__(self, training_set, initial_budget):
        super().__init__(training_set, initial_budget)

    def _train(self, training_set):
        """TODO: explore data (use class DataExploration) to determine a reasonable range"""
        # training_set.data_feature
        pass

    def _bidding_function(self, utility=None, cost=None):
        """Deploy the learned bidding model"""
        # write fwd step of nn
        pass
