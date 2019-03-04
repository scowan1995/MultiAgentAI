import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


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

    def format_data(self, data, target=None, split_useragent=True):
        """
        Formats the data by splitting user agent into os and browser
        makes the usertag list into catagories
        drops columns that can't be used in prediction
        converts non numerics to catagories
        """
        data = shuffle(data)
        if split_useragent:
            data = self.split_useragent_feature(data)
        data = self.catagorise_usertag(data)

        X = self.get_input_data(data)
        X = self.catagorise_numerics(X, split_useragent)
        Y = data.loc[:, target]

        if self.debug_flag:
            print("X:")
            print(X)
            print("Y:")
            print(Y)

        return X, Y

    def downsample_data(self, X, Y):
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

    def _get_negatives(self, x, y, pos_clicksX, neg_clicksX, neg_clicksY):
        ix = np.random.permutation(
            pos_clicksX.index)[:pos_clicksX.shape[0]]
        negsX = neg_clicksX.iloc[ix]
        negsY = neg_clicksY.iloc[ix]
        return negsX, negsY
