from Preprocessing.single_set import SingleSet
from Bidders.basic_bidding_agent import BasicBiddingAgent
from Main.plot_utils import *

import numpy as np
from sklearn.linear_model import LogisticRegression
import scipy.stats as stats


class BudgetAwareLogisticRegressionBiddingAgent(BasicBiddingAgent):
    def __init__(self, training_set, initial_budget, campaign_duration=None):
        self._campaign_budget = initial_budget
        self._campaign_duration = campaign_duration
        self._price_market_upper_bound = 0
        self._fitted_marketprice_distribution = None

        self._current_features = None
        self._features_to_drop = ['userid', 'urlid', 'bidid', 'domain', 'region', 'url', 'keypage', 'city']

        self._logistic_regressor = None

        super().__init__(training_set, initial_budget)

    def _train(self, training_set):
        self._compute_price_market_upper_bound(training_set)
        self._train_click_probability_predictor(training_set)

    def _process_bid_request(self, ad_user_auction_info=None):
        current_features = SingleSet.drop_features_from_single_row(ad_user_auction_info,
                                                                   self._features_to_drop)
        self._current_features = np.asarray(current_features)
        self._current_slot_price = current_features['slotprice']
        self._bid_value = self._bidding_function()

    def _bidding_function(self):
        if self._campaign_duration is None:
            raise ValueError
        r = self._click_probability()
        # b = self._campaign_budget
        b = self._current_budget
        l = self._price_market_upper_bound
        # t = self._campaign_duration
        t = self._campaign_duration - self._placed_bids
        return 2 * r[0][1] * ((b * l**2) / t)**(1/3)

    def _train_click_probability_predictor(self, training_set):
        training_features = training_set
        if self._features_to_drop is not None:
            training_features = training_set.drop_features(self._features_to_drop)
        training_click_labels = np.asarray(training_set.data_targets['click'])

        class_weight = {0: (1793 / 2429188), 1: (1 - (1793 / 2429188))}
        self._logistic_regressor = LogisticRegression(class_weight=class_weight, C=1.0, penalty='l2', verbose=10)
        self._logistic_regressor.fit(training_features, training_click_labels)

        print(self._logistic_regressor.coef_)
        print("\n-- logistic regression completed --")

    def _click_probability(self):
        single_feature_vector = self._current_features.reshape(1, -1)
        click_predictions_prob = self._logistic_regressor.predict_proba(single_feature_vector)
        return click_predictions_prob

    def _compute_price_market_upper_bound(self, training_set):
        max_payprice = training_set.data_targets['payprice'].max()
        # diff = training_set.data_targets['bidprice'] - training_set.data_targets['payprice']
        # epsilon = diff.mean()
        # max_bidprice = training_set.data_targets.loc[lambda x: x["click"] == 1, "payprice"]
        self._price_market_upper_bound = max_payprice  # + epsilon

    def fit_marketprice_gamma_distribution(self, training_set):
        market_price = np.asarray(training_set.data_targets['payprice'])
        min_market_price = training_set.data_targets['payprice'].min()
        max_market_price = training_set.data_targets['payprice'].max()

        shape, loc, scale = stats.gamma.fit(market_price)
        self._fitted_marketprice_distribution = lambda x: stats.gamma.pdf(x=x, a=shape, loc=loc, scale=scale)

        x_gamma = np.linspace(min_market_price, max_market_price, 100)
        y_gamma = self._fitted_marketprice_distribution(x_gamma)

        plot_distribution(market_price, x_gamma, y_gamma, "marketprice_distribution")
        return self._fitted_marketprice_distribution, min_market_price, max_market_price

    def fit_marketprice_log_normal_distribution(self, training_set, plot=True):
        market_price = np.asarray(training_set.data_targets['payprice'])
        min_market_price = training_set.data_targets['payprice'].min()
        max_market_price = training_set.data_targets['payprice'].max()

        shape, loc, scale = stats.lognorm.fit(market_price)
        self._fitted_marketprice_distribution = lambda x: stats.lognorm.pdf(x=x, a=shape, loc=loc, scale=scale)

        x_lognorm = np.linspace(min_market_price, max_market_price, 100)
        y_lognorm = self._fitted_marketprice_distribution(x_lognorm)

        if plot:
            plot_distribution(market_price, x_lognorm, y_lognorm, "marketprice_distribution")
        return self._fitted_marketprice_distribution, min_market_price, max_market_price

    def set_campaign_duration_from_set(self, campaign_set):
        self._campaign_duration = len(campaign_set.data.index)

    def set_features_to_drop(self, drop):
        self._features_to_drop = drop

