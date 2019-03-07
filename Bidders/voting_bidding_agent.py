from sklearn.ensemble import (
    GradientBoostingRegressor,
    VotingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import sys
import os

from basic_bidding_agent import BasicBiddingAgent
from bidder_utils import BidderUtils


class EnsembleBiddingAgent(BasicBiddingAgent):
    def __init__(self, training_set, initial_budget, estimators):
        self.estimators = estimators
        self.utils = BidderUtils()
        self.click_model = None
        self.pay_model = GradientBoostingRegressor(n_estimators=3000)

        super().__init__(training_set, initial_budget)

    def _train(self, training_set):
        print("training...")
        x_clicks, y_clicks = self.utils.format_data(training_set.copy(), target="click")
        x_clicks, y_clicks = self.utils.downsample_data(x_clicks, y_clicks)
        #x_clicks, self.xclicks_val, y_clicks, self.y_clicks_val = train_test_split(
        #    x_clicks, y_clicks, test_size=0.15
        #)
        x_pay, y_pay = self.utils.format_data(training_set.copy(), target="payprice")
        x_pay, y_pay = self.utils.downsample_data(x_pay, y_pay)
        print("Y pay", y_pay)
        #x_pay, self.xpay_val, y_pay, self.y_pay_val = train_test_split(
        #    x_pay, y_pay, test_size=0.1
        #)
        self.click_model = VotingClassifier(self.estimators, voting="soft", n_jobs=-1)
        self.click_model.fit(x_clicks, y_clicks)
        self.pay_model.fit(x_pay, y_pay)

    def test(self, test, test_y_click, test_y_payprice, cols):
        print("testing")
        test_x = test.loc[:, cols]
        print("Click score", self.click_model.score(test_x, test_y_click))
        print("Payprice score", self.pay_model.score(test_x, test_y_payprice))
        test_modelled_click = self.click_model.predict_proba(test_x)[:, 1]
        test_modelled_pay = self.pay_model.predict(test_x)
        test_moddelled = pd.concat([pd.DataFrame(test_modelled_click), pd.DataFrame(test_modelled_pay)], axis=1)
        test_moddelled.to_csv("Data/saved/test_set_modelled.csv")
        return test_moddelled
        
    def create_val_preds(self, val):
        val_click = self.click_model.predict_proba(val)[:, 1]
        val_pay = self.pay_model.predict(val)
        val_modelled = pd.concat([pd.DataFrame(val_click), pd.DataFrame(val_pay)], axis=1)
        val_modelled.to_csv("Data/saved/Validation_modelled.csv")
        return val_modelled


    def _bidding_function(self, utility=None, cost=None, x=None):
        is_click = self.click_model.predic_proba(x) > 0.3
        pay_prediction = self.pay_model.predict(x)
        if is_click:
            return pay_prediction
        else:
            return 0


def get_training_set():
    data_path = os.path.abspath(os.pardir + "/MultiAgentAI/Data/train_small.csv")
    test_path = os.path.abspath(os.pardir + "/MultiAgentAI/Data/test_small.csv")
    val_path = os.path.abspath(os.pardir + "/MultiAgentAI/Data/validation.csv")
    train = pd.read_csv(data_path, na_values=["Na", "null"]).fillna(0)
    test = pd.read_csv(test_path, na_values=["Na", "null"]).fillna(0)
    val = pd.read_csv(val_path, na_values=["Na", "null"]).fillna(0)
    return train, val, test


if __name__ == "__main__":
    utils = BidderUtils()
    train, test, val = get_training_set()
    """
    fx, fy = utils.format_data(train, target="click", shuff=False)
    trainx, trainy = utils.downsample_data(fx, fy)
    val = utils.format_data(val, shuf=False)
    notinVal = [i for i in trainx.columns if i not in val.columns]
    notinTrain = [i for i in val.columns if i not in trainx.columns]
    print("Any columns not in val or train:")
    print(notinVal)
    print(notinTrain)
    """
    val = utils.format_data(val, shuf=False)
    test, test_click, test_pay = utils.format_data(test, target=["click", "payprice"], shuf=False)
    print("test", test.shape, " cols", test.columns)
    print("test clcik", test_click.shape)
    print("test_pay", test_pay.shape)

    print("svm")
    svm = SVC(kernel="poly", cache_size=1024, probability=True, class_weight="balanced", max_iter=500)

    print("logistic regression")
    lr = LogisticRegression(class_weight="balanced", max_iter=100)

    print("random forest")
    rf = RandomForestClassifier(n_estimators=3000, class_weight="balanced", n_jobs=-1)

    e = EnsembleBiddingAgent(
        train,
        6250 * 1000,
        list([("logistic classifier", lr), ("SVM", svm), ("Random Forest", rf)]),
    )
    e.test(test, test_click, test_pay, val.columns)
    e.create_val_preds(val)
