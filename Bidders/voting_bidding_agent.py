from basic_bidding_agent import BiddingAgent
from sklearn.ensemble import VotingClassifier, GradientBoostingRegressor


class EnsembleBiddingAgent(BiddingAgent):

    def __init__(self, training_set, initial_budget, click_models):
        self.estimators = click_models
        self.click_model = None
        self.pay_model = GradientBoostingRegressor()

        super().__init__(training_set, initial_budget)

    def _train(self, training_set):
        trainX, trainY = self.downsample_training_set()
        self.click_model = VotingClassifier(self.estimators, n_jobs=-1)
        self.click_model.fit(trainX, trainY)
        self.pay_model.fit(trainX, trainY)

    def _bidding_function(self, utility=None, cost=None, x=None):
        is_click = self.click_model.predict(x)
        pay_prediction = self.pay_model.predict(x)
        if is_click:
            return pay_prediction
        else:
            return 0

    def format_data(self, data):
        pass

    def downsample_training_set(self):
        pass
