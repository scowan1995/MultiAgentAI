import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from datetime import datetime


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

    def test(self, min_prob, multiplier, budget, max_prob=None):
        # print("model score:", self.model.score(self.testx, self.testy))
        predictions = self.model.predict(self.testx)
        predictions *= multiplier
        clicks = 0
        for i in range(predictions.shape[0]):
            if (
                predictions[i] >= self.testy.iloc[i]
                and (self.testx["click_prob"].iloc[i] > min_prob)
                and (max_prob is None or max_prob > self.testx["click_prob"].iloc[i])
            ):
                budget -= self.testy.iloc[i]
                if budget < 0:
                    break
                if self.test_clicks.iloc[i] > 0:
                    clicks += 1
        local_prob = min_prob
        more_clicks = [0]
        while budget > 100 and local_prob > 0.001:
            print(
                budget,
                "leftover when using multiplier:",
                multiplier,
                "and min prob:",
                local_prob,
                ". Continuing with min prob",
                str(local_prob * 0.8),
            )
            c, p, b = self.test(
                local_prob * 0.9, multiplier, budget, max_prob=local_prob
            )
            local_prob = p
            budget = b
            more_clicks.append(c)
        if max_prob is None:
            with open("results.csv", "a") as f:
                f.write(
                    str(
                        str(clicks)
                        + " , "
                        + str(budget)
                        + " , "
                        + str(min_prob)
                        + " , "
                        + str(multiplier) + "\n"
                    )
                )
            # print("total clicks", clicks + sum(more_clicks))
            # print("have budget", budget)
            # print("using min", min_prob)
            # print("using mul", multiplier)
        return clicks, min_prob, budget

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
    initial_budget = 6250 * 1000
    range_probs = np.random.uniform(0.0001, 0.90, size=40)
    pay_muls = np.append(np.random.uniform(0.5, 3, size=12), 1)
    i = 0
    total = 40 * 12
    best = 0
    best_bound = 0
    best_mul = 0
    final_budg = 0
    start = datetime.now()
    with open("results.csv", "w") as f:
        f.write("click, budget, min_prob, multiplier \n")
    for multiplier in pay_muls:
        for prob in range_probs:
            i += 1
            clicks, bound, budget = lb.test(prob, multiplier, initial_budget)
            if clicks > best:
                best_mul = multiplier
                best = clicks
                best_bound = bound
                final_budg = budget
            if i % 10 == 0:
                print(str(i * 100 / total), "% way through")
                print("best click", best)
                print(
                    "its been ",
                    datetime.now() - start,
                    " hours, mins, seconds since hyper param tuning began",
                )
    print(
        "Best =",
        best,
        "with bound",
        best_bound,
        "and best multiplier",
        best_mul,
        "with",
        final_budg,
        "remaining",
    )

