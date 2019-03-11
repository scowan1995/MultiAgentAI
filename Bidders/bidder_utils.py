import numpy as np
import pandas as pd
from sklearn.utils import shuffle


class BidderUtils:
    debug_flag = False

    def __init__(self):
        pass

    def split_useragent_feature(self, data):
        data[["os", "browser"]] = data.apply(
            lambda x: x["useragent"].split("_"), axis=1, result_type="expand"
        )
        data.drop(columns=["useragent"], inplace=True)
        return data

    def catagorise_usertag(self, data):
        data = data.drop("usertag", 1).join(
            data.usertag.str.join("|").str.get_dummies()
        )
        return data

    def get_input_data(self, data):
        unusable_cols = ["bidprice", "click", "payprice", "urlid", "bidid"]
        input_cols = [x for x in data.columns if x not in unusable_cols]
        return data.loc[:, input_cols]

    def catagorise_numerics(self, data, split_useragent=True):
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
            catagories.extend(["usertag", "useragent"])
        for cat in catagories:
            data[cat] = data[cat].astype("category").cat.codes
        return data

    def format_data(self, data, target=None, split_useragent=True, shuf=False):
        """
        Formats the data by splitting user agent into os and browser
        makes the usertag list into catagories
        drops columns that can't be used in prediction
        converts non numerics to catagories
        """
        if shuf:
            data = shuffle(data)
        if split_useragent:
            data = self.split_useragent_feature(data)
        data = self.catagorise_usertag(data)

        X = self.get_input_data(data)
        X = self.catagorise_numerics(X, split_useragent)
        Y = None
        if target is not None and type(target) is str:
            Y = data.loc[:, target]
            print("GOt Y")
            print(Y)
        elif target is not None:
            ys = [0, 0]
            for i in range(len(target)):
                ys[i] = data.loc[:, target[i]]
            Y = ys

        if self.debug_flag:
            print("X:")
            print(X)
            print("Y:")
            print(Y)
        if Y is None:
            return self.normalized_df(X)
        if type(target) is not str:
            return self.normalized_df(X), Y[0], Y[1]

        return self.normalized_df(X), Y

    def downsample_data(self, X, Y):
        """
        Downsample the data so we have equal clicks, no clicks
        """
        positive_clicksX = X[(Y == 1)]
        positive_clicksY = Y[(Y == 1)]
        negative_clicksX = X[(Y == 0)]
        negative_clicksY = Y[(Y == 0)]
        negX, negY = self._get_negatives(
            X, Y, positive_clicksX, negative_clicksX, negative_clicksY
        )
        x = positive_clicksX.append(negX)
        y = positive_clicksY.append(negY)
        return x, y

    def normalized_df(self, df: pd.DataFrame):
        return (df - df.min()) / (df.max() - df.min())

    def _get_negatives(self, x, y, pos_clicksX, neg_clicksX, neg_clicksY):
        ixs = np.random.permutation(neg_clicksX.index)[:pos_clicksX.shape[0]]

        for i in range(len(ixs)):
            if ixs[i] >= neg_clicksX.shape[0]:
                ixs[i] = 0
        print(pos_clicksX)
        print("pos clicks shape", pos_clicksX.shape)
        print("ixs", ixs.shape)
        print("neg click x", neg_clicksX.shape)
        print("neg click y", neg_clicksY.shape)
        negsX = neg_clicksX.iloc[ixs]
        negsY = neg_clicksY.iloc[ixs]
        return negsX, negsY
