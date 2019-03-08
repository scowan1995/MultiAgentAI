import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


class LinearRegressionBidder:
    def __init__(self):
        self.train_set, self.test_set, self.val = self.get_data()
        self.X = self.train_set.loc[:, ["click_prob", "pay_estimate", "click"]]
        self.Y = self.train_set.loc[:, "payprice"]
        self.trainx, self.testx, self.trainy, self.testy = train_test_split(
            self.X, self.Y, test_size=0.1
        )
        self.train_clicks = self.trainx.loc[:, "click"]
        self.trainx.drop("click", axis=1, inplace=True)
        self.test_clicks = self.testx.loc[:, "click"]
        self.testx.drop("click", axis=1, inplace=True)
        self.model = LinearRegression(normalize=True, n_jobs=-1)

    def train(self):
        return self.model.fit(self.trainx, self.trainy)

    def test(self, min_prob, multiplier):
        budget = 6520 * 1000
        # print("model score:", self.model.score(self.testx, self.testy))
        predictions = self.model.predict(self.testx)
        predictions *= multiplier
        clicks = 0
        for i in range(predictions.shape[0]):
            if predictions[i] >= self.testy.iloc[i] and (
                self.testx["click_prob"].iloc[i] > min_prob
            ):
                budget -= self.testy.iloc[i]
                if budget < 0:
                    break
                if self.test_clicks.iloc[i] > 0:
                    clicks += 1
        print("total clicks", clicks)
        print("have budget", budget)
        print("using min", min_prob)
        print("using mul", multiplier)
        return clicks, min_prob

    def get_data(self):
        train = pd.read_csv("Data/saved/Train_modelled.csv")
        train_clicks = pd.read_csv("Data/train.csv").loc[:, "click"]
        train = pd.concat([train, train_clicks], axis=1)
        test = pd.read_csv("Data/saved/test_set_modelled.csv")
        test_clicks = pd.read_csv("Data/validation.csv").loc[:, "click"]
        test = pd.concat([test, test_clicks], axis=1)
        val = pd.read_csv("Data/saved/Validation_modelled.csv")
        print("train size", train.shape)
        print("test size", test.shape)
        print("val size", val.shape)
        return train, test, val


if __name__ == "__main__":
    lb = LinearRegressionBidder()
    print("training")
    lb.train()
    print("testing")
    range_probs = np.random.uniform(0.0001, 0.95, size=100)
    pay_muls = np.append(np.random.uniform(0.5, 5, size=35), 1)
    best = 0
    best_bound = 0
    best_mul = 0
    for multiplier in pay_muls:
        for prob in range_probs:
            clicks, bound = lb.test(prob, multiplier)
            if clicks > best:
                best_mul = multiplier
                best = clicks
                best_bound = bound
    print("Best =", best, "with bound", best_bound, "and best multiplier", best_mul)

