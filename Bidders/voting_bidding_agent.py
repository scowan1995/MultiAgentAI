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

<<<<<<< HEAD
from basic_bidding_agent import BasicBiddingAgent
from bidder_utils import BidderUtils


class EnsembleBiddingAgent(BasicBiddingAgent):
    def __init__(self, training_set, initial_budget, estimators):
        self.estimators = estimators
=======
from .basic_bidding_agent import BasicBiddingAgent
from .bidder_utils import BidderUtils


class EnsembleBiddingAgent(BasicBiddingAgent):

    def __init__(self, training_set, initial_budget, click_models):
        self.estimators = [LogisticRegression(class_weight="balanced", max_iter=500)]
>>>>>>> b6c46cec8cfb1d2aa8ccff09f5a274d2e513a21b
        self.utils = BidderUtils()
        self.click_model = None
        self.pay_model = GradientBoostingRegressor()

        super().__init__(training_set, initial_budget)

    def _train(self, training_set):
        print("training...")
        x_clicks, y_clicks = self.utils.format_data(training_set, target="click")
        x_clicks, y_clicks = self.utils.downsample_data(x_clicks, y_clicks)
        x_clicks, self.xclicks_val, y_clicks, self.y_clicks_val = train_test_split(
            x_clicks, y_clicks, test_size=0.15
        )
        x_pay, y_pay = self.utils.format_data(training_set, target="payprice")
        x_pay, self.xpay_val, y_pay, self.y_pay_val = train_test_split(
            x_pay, y_pay, test_size=0.1
        )
        x_clicks.to_csv("./Data/xclicks.csv", index=False)
        y_clicks.to_csv("./Data/y_clicks.csv", index=False)
        self.xclicks_val("./Data/xclicksval.csv", index=False)
        self.y_clicks_val("./Data/yclicksval.csv", index=False)
        x_pay.to_csv("./Data/xpay.csv", index=False)
        y_pay.to_csv("./Data/y_pay.csv", index=False)
        self.xpay_val("./Data/xpayval.csv", index=False)
        self.y_pay_val("./Data/ypayval.csv", index=False)
        print("Data saved")

        self.click_model = VotingClassifier(self.estimators, voting="soft", n_jobs=-1)
        self.click_model.fit(x_clicks, y_clicks)
        self.pay_model.fit(x_pay, y_pay)

    def test(self, val):
        print("testing...")
        res = self.click_model.predict(val)
        np.savetxt("click_predictions.csv", res, delimiter=",")
        res = self.pay_model.predict(val).multiply(1.5)
        np.savetxt("pay_predictions.csv", res, delimiter=",")
        return (
            self.click_model.score(self.xclicks_val, self.y_clicks_val),
            self.pay_model.score(self.xpay_val, self.y_pay_val),
        )

    def _bidding_function(self, utility=None, cost=None, x=None):
        is_click = self.click_model.predic_proba(x) > 0.3
        pay_prediction = self.pay_model.predict(x)
        if is_click:
            return pay_prediction
        else:
            return 0


def get_training_set():
    data_path = os.path.abspath(os.pardir + "/MultiAgentAI/Data/train.csv")
    val_path = os.path.abspath(os.pardir + "/MultiAgentAI/Data/validation.csv")
    train = pd.read_csv(data_path, na_values=["Na", "null"]).fillna(0)
    val = pd.read_csv(val_path, na_values=["Na", "null"]).fillna(0)
    return train, val


if __name__ == "__main__":
    utils = BidderUtils()
    train, val = get_training_set()
    t = train.copy()
    v = val.copy()
    print("data copied")
    fx, fy = utils.format_data(t, target="click")
    x, y = utils.downsample_data(fx, fy)
    x, y = shuffle(x, y)
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.15)
    x.to_csv("Data/x.csv", index=False)
    y.to_csv("Data/y.csv", index=False)
    xtrain.to_csv('./Data/xtrain.csv', index=False)
    xtest.to_csv('./Data/xtest.csv', index=False)
    ytrain.to_csv('./Data/ytrain.csv', index=False)
    ytest.to_csv('./Data/test_saved.csv', index=False)
    valx = utils.format_data(v)
    print("column matching:")
    val_formatted = utils.format_data(val, shuf=False)
    val_formatted.to_csv('./Data/val_formatted.csv', index=False)
    notinVal = [i for i in x.columns if i not in val_formatted.columns]
    notinTrain = [i for i in val_formatted.columns if i not in x.columns]
    print(notinVal)
    print(notinTrain)

    print("svm")
    svm = SVC(kernel="poly", cache_size=1024, class_weight="balanced", max_iter=50)
    svm.fit(x, y)
    print("svm score:", svm.score(xtest, ytest))

    print("logistic regression")
    lr = LogisticRegression(class_weight="balanced", max_iter=50)
    lr.fit(x, y)
    score = lr.score(xtest, ytest)
    print("lr score:", score)

    print("random forest")
    rf = RandomForestClassifier(n_estimators=10, class_weight="balanced", n_jobs=-1)
    rf.fit(x, y)
    print("RF score:", rf.score(xtest, ytest))

    e = EnsembleBiddingAgent(
        train,
        6250 * 1000,
        list([("logistic classifier", lr), ("SVM", svm), ("Random Forest", rf)]),
    )
    print(e.test(val_formatted))
