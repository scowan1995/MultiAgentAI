from sklearn import tree
from sklearn.svm import SVR
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from datetime import datetime
import os


class Bidder:
    # TODO: tune over max size, min sample, coefficients and possibly others

    def __init__(self, algorithm, dont_include):
        # don't include is used to specify features to ignore
        # This is to help see if any feature is an outsized predictor
        self.dont_include = dont_include
        self.trainX, self.testX, self.trainY, self.testY = self.get_data()
        self.model = algorithm

    def get_data(self):
        data_path = os.path.abspath(os.pardir + "/MultiAgentAI/Data/train.csv")
        data = pd.read_csv(data_path, na_values=["Na", "null"]).fillna(0)
        data = shuffle(data)

        # these columns are not used because they are to be predicted or are
        # user specific and so any correlation will be spurrious
        unusable_cols = ["bidprice", "click", "payprice", "userid", "urlid", "bidid"]
        if self.dont_include is not None:
            unusable_cols.append(self.dont_include)

        input_cols = [x for x in list(data.columns) if x not in unusable_cols]
        X = data.loc[:, input_cols]
        catagories = [
            "useragent",
            "IP",
            "domain",
            "url",
            "slotid",
            "slotvisibility",
            "creative",
            "keypage",
            "usertag",
        ]
        if self.dont_include is not None and self.dont_include in catagories:
            catagories.remove(self.dont_include)

        for cat in catagories:
            X[cat] = X[cat].astype("category").cat.codes
        Y = data.loc[:, "bidprice"]
        _trainX, _testX, _trainY, _testY = train_test_split(X, Y, test_size=0.2)
        return _trainX, _testX, _trainY, _testY

    def train(self):
        self.model.fit(self.trainX, self.trainY)

    def test(self):
        score = self.model.score(self.testX, self.testY)
        print("Achieved score:")
        print(score)
        preds = self.model.predict(self.testX)
        x = np.divide(np.sum(np.square(self.testY - preds)), self.testX.shape[0])
        print("MSE:", x)
        print()


if __name__ == "__main__":
    svm = SVR()
    dtree = tree.DecisionTreeRegressor()
    models = [svm, dtree]
    remaining = [
        "weekday",
        "hour",
        "useragent",
        "IP",
        "region",
        "city",
        "adexchange",
        "domain",
        "url",
        "slotid",
        "slotwidth",
        "slotheight",
        "slotvisibility",
        "slotformat",
        "slotprice",
        "creative",
        "keypage",
        "advertiser",
        "usertag",
    ]
    for m in models:
        print("using", m)
        d = Bidder(m, None)
        d.train()
        print("testing")
        print(datetime.now().time())
        d.test()
