from sklearn import tree, svm
from sklearn.naive_bayes import GaussianNB
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import roc_auc_score
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    VotingClassifier,
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
)
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from datetime import datetime
import os
from joblib import dump, load


class Bidder:
    def __init__(
        self, algorithm, dont_include, ensemble_creator=True, use_kernel=False
    ):
        print("in constructor for", algorithm)
        self.use_kernel = use_kernel
        # don't include is used to specify features to ignore
        # This is to help see if any feature is an outsized predictor
        self.dont_include = dont_include
        dataX, self.testX, dataY, self.testY = self.get_data()
        self.positive_clicksX = dataX[(dataY == 1)]
        self.negative_clicksX = dataX[(dataY == 0)]
        self.positive_clicksY = dataY[(dataY == 1)]
        self.negative_clicksY = dataY[(dataY == 0)]
        if ensemble_creator:
            negX, negY = self.get_data_chunk()
            print("posx", self.positive_clicksX.shape)
            print("negx", negX.shape)
            self.trainX = self.positive_clicksX.append(negX)
            print("train shape using small batch", self.trainX.shape)
            self.trainY = self.positive_clicksY.append(negY)
            self.trainX, self.trainY = shuffle(self.trainX, self.trainY)
        else:
            self.trainX, self.trainY = dataX, dataY
            print("train shape using full", self.trainX.shape)

        # gaussian kernal because non linear
        if use_kernel:
            self.rbf_feature = RBFSampler(gamma=2, random_state=1)
            self.trainX = self.rbf_feature.fit_transform(self.trainX, self.trainY)
            self.testX = self.rbf_feature.transform(self.testX)
            print("train shape after kernel", self.trainX.shape)

        self.model = algorithm[1]
        self.modelname = algorithm[0]

    def get_data_chunk(self):
        ix = np.random.permutation(self.positive_clicksX.index)[
            : self.positive_clicksX.shape[0]
        ]
        negsX = self.negative_clicksX.iloc[ix]
        negsY = self.negative_clicksY.iloc[ix]
        return negsX, negsY

    def get_validation(self, data_path, split_useragent=True):
        data = pd.read_csv(data_path, na_values=["Na", "null"]).fillna(0)
        data = shuffle(data)
        if split_useragent:
            split_UA = lambda x: x["useragent"].split("_")
            data[["os", "browser"]] = data.apply(split_UA, axis=1, result_type="expand")
            data.drop(columns=["useragent"], inplace=True)
            data = data.drop("usertag", 1).join(
                data.usertag.str.join("|").str.get_dummies()
            )
        unusable_cols = ["bidprice", "click", "payprice", "urlid", "bidid"]
        if self.dont_include is not None:
            unusable_cols.append(self.dont_include)

        input_cols = [x for x in list(data.columns) if x not in unusable_cols]
        X = data.loc[:, input_cols]
        catagories = [
            "IP",
            "domain",
            "url",
            "slotid",
            "userid",
            "slotvisibility",
            "creative",
            "keypage",
        ]
        if split_useragent:
            catagories.extend(["os", "browser"])
        else:
            catagories.extent(["usertag", "useragent"])
        if self.dont_include is not None and self.dont_include in catagories:
            catagories.remove(self.dont_include)

        for cat in catagories:
            X[cat] = X[cat].astype("category").cat.codes
        # _trainX, _trainY = train_test_split(X, Y, test_size=0)
        if self.use_kernel:
            X = self.rbf_feature.transform(X)
        return X

    def get_data(self, target="click"):
        data_path = os.path.abspath(os.pardir + "/MultiAgentAI/Data/train.csv")
        val_path = os.path.abspath(os.pardir + "/MultiAgentAI/Data/validation.csv")
        _trainX, _trainY = self.format_data(data_path, target, split_useragent=True)
        _valX, _valY = self.format_data(val_path, target, split_useragent=True)
        # print("train shapes: x:", _trainX.shape, "Y: ", _trainY.shape)
        return _trainX, _valX, _trainY, _valY

    def format_data(self, data_path, target, split_useragent=False):
        data = pd.read_csv(data_path, na_values=["Na", "null"]).fillna(0)
        data = shuffle(data)
        if split_useragent:
            split_UA = lambda x: x["useragent"].split("_")
            data[["os", "browser"]] = data.apply(split_UA, axis=1, result_type="expand")
            data.drop(columns=["useragent"], inplace=True)
            mlb = MultiLabelBinarizer()
            data = data.drop("usertag", 1).join(
                data.usertag.str.join("|").str.get_dummies()
            )
            # print(type(split_pieces))
        # data["browser"] = pd.Series(split_pieces[1], index=data.index)
        # data["os"] = pd.Series(split_pieces[0], index=data.index)

        # these columns are not used because they are to be predicted or are
        # user specific and so any correlation will be spurrious
        unusable_cols = ["bidprice", "click", "payprice", "urlid", "bidid"]
        if self.dont_include is not None:
            unusable_cols.append(self.dont_include)
        if target not in unusable_cols:
            unusable_cols.append(target)

        input_cols = [x for x in list(data.columns) if x not in unusable_cols]
        X = data.loc[:, input_cols]
        catagories = [
            "IP",
            "domain",
            "url",
            "slotid",
            "userid",
            "slotvisibility",
            "creative",
            "keypage",
        ]
        if split_useragent:
            catagories.extend(["os", "browser"])
        else:
            catagories.extent(["usertag", "useragent"])
        if self.dont_include is not None and self.dont_include in catagories:
            catagories.remove(self.dont_include)

        for cat in catagories:
            X[cat] = X[cat].astype("category").cat.codes
        Y = data.loc[:, target]
        # _trainX, _trainY = train_test_split(X, Y, test_size=0)
        return X, Y

    def get_data(self, target="click"):
        data_path = os.path.abspath(os.pardir + "/MultiAgentAI/Data/train.csv")
        val_path = os.path.abspath(os.pardir + "/MultiAgentAI/Data/validation.csv")
        _trainX, _trainY = self.format_data(data_path, target, split_useragent=True)
        _valX, _valY = self.format_data(val_path, target, split_useragent=True)
        # print("train shapes: x:", _trainX.shape, "Y: ", _trainY.shape)
        return _trainX, _valX, _trainY, _valY

    def train(self):
        self.model.fit(self.trainX, self.trainY)
        return self.model

    def test(self):
        score = self.model.score(self.testX, self.testY)
        print("score", score)
        print("score from guessing all zeros in the validation set:")
        print(1 - (np.sum(self.testY) / self.testY.shape[0]))
        print("ROC Score:")
        # print("train shape", self.trainX.shape)
        #  print("test  shape", self.testX.shape)
        preds = self.model.predict(self.testX)
        print("number of 1s:", sum(preds))
        # dump(self.model, self.modelname + "onehot")
        print(roc_auc_score(self.testY, preds))
        val = self.get_validation(os.pardir + "/MultiAgentAI/Data/validation.csv")
        res = self.model.predict(val)
        np.savetxt("predictions.csv", res, delimiter=',')



if __name__ == "__main__":
    """
    gbr = GradientBoostingClassifier()
    gbr_min5 = GradientBoostingClassifier(min_samples_leaf=5)
    gbr_min15 = GradientBoostingClassifier(min_samples_leaf=15)
    dtree3 = tree.DecisionTreeClassifier(min_samples_leaf=3)
    dtree5 = tree.DecisionTreeClassifier(min_samples_leaf=5)
    dtree9 = tree.DecisionTreeClassifier(min_samples_leaf=9)
    models = [
      #  ("gbr_min15_click", gbr_min15),
      #  ("gbr_min5_click", gbr_min5),
        ("gbr_click", gbr),
      #  ("dTree3_click", dtree3),
        ("dTree5_click", dtree5)]
       # ("dTree9_click", dtree9)]
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
    """
    estimators = []
    num_estimators = 11
    models_already_created = False
    do_LR = True
    if models_already_created:
        for i in range(num_estimators):
            model = load("svm" + str(i))
            estimators.append(("svm" + str(i), model))
        v = VotingClassifier(estimators, n_jobs=-1)
        voting_bidder = Bidder(("voting", v), None)
        voting_bidder.train()
        voting_bidder.test()
    elif do_LR:
        """
        kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        clf3 = GaussianProcessClassifier(
            kernel=kernel,
            max_iter_predict=100,
            multi_class="one_vs_one",
            n_jobs=-1,
            n_restarts_optimizer=5,
        )
        b5 = Bidder(("gp", clf3), None)
        model5 = b5.train()
        print("gaussian process")
        b5.test()
        estimators.append(("gaussian process", model5))
        """
        model = LogisticRegression(class_weight="balanced", max_iter=500)
        b = Bidder(("LR_balanced", model), None, False, use_kernel=True)
        model = b.train()
        print("LR balanced with rdf kernel")
        b.test()
        estimators.append(("LR kernal", model))

        model1 = LogisticRegression(class_weight="balanced", max_iter=500)
        b1 = Bidder(("LR_balanced", model), None, False, use_kernel=False)
        model1 = b1.train()
        print("LR no kernal")
        b1.test()
        estimators.append(("LR", model))

        print("random forrest class")
        gbc = RandomForestClassifier(
            n_estimators=1000, class_weight="balanced", n_jobs=-1
        )
        b4 = Bidder(("RFC", gbc), None)
        model4 = b4.train()
        print("RFC test")
        b4.test()
        estimators.append(("RFC", model4))

        v = VotingClassifier(estimators, n_jobs=-1)
        voting_bidder = Bidder(("voting", v), None)
        voting_bidder.train()
        print("voting")
        voting_bidder.test()

    else:
        for n in range(num_estimators):
            np.random.seed(seed=n)
            print("creating estimator", n)
            dtree = svm.SVC(C=10, tol=0.0001, gamma="scale")
            #          LogisticRegression(solver='liblinear', max_iter=500, C=100, tol=0.00001)
            #     )  # tree.DecisionTreeClassifier(min_samples_leaf=n+1)
            name_model = ("svm" + str(n), dtree)
            d = Bidder(name_model, None)
            model = d.train()
            d.test()
            estimators.append((name_model[0], model))
        v = VotingClassifier(estimators, n_jobs=-1)
        voting_bidder = Bidder(("voting", v), None, False)
        voting_bidder.train()
        voting_bidder.test()

    # for m in models:
    #    print("using", m[0])
    #    d = Bidder(m, None)
    #    d.train()
    #    d.test()
