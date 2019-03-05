from sklearn.ensemble import GradientBoostingRegressor, VotingClassifier
from sklearn.linear_model import LogisticRegression

from .basic_bidding_agent import BasicBiddingAgent
from .bidder_utils import BidderUtils


class EnsembleBiddingAgent(BasicBiddingAgent):

    def __init__(self, training_set, initial_budget, click_models):
        self.estimators = [LogisticRegression(class_weight="balanced", max_iter=500)]
        self.utils = BidderUtils()
        self.click_model = None
        self.pay_model = GradientBoostingRegressor()

        super().__init__(training_set, initial_budget)

    def _train(self, training_set):
        x_clicks, y_clicks = self.utils.format_data(training_set, target="click")
        x_clicks, y_clicks = self.utils.downsample(x_clicks, y_clicks)
        x_pay, y_pay = self.utils.format_data(training_set, target="payprice")
        self.click_model = VotingClassifier(self.estimators, n_jobs=-1)
        self.click_model.fit(x_clicks, y_clicks)
        self.pay_model.fit(x_pay, y_pay)

    def _bidding_function(self, utility=None, cost=None, x=None):
        is_click = self.click_model.predict(x)
        pay_prediction = self.pay_model.predict(x)
        if is_click:
            return pay_prediction
        else:
            return 0
