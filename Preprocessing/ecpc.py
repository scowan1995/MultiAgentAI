import os
import pandas as pd
import matplotlib.pylab as plt

from Configs.configs import configs


class ECPC:
    # Class for calculating the expected cost per click

    def __init__(self):
        path = os.path.abspath(__file__ + "/../") + "/Data/train.csv"
        data_pandas = pd.read_csv(path, na_values=["Na", "null"]).fillna(0)
        self.data = data_pandas
        self.data_targets = self.data[configs["target"]]
        self.plottable_items = []
        self.feature_list = [
            "weekday",
            "hour",
            "useragent",
            "region",
            "city",
            "adexchange",
            "creative",
            "advertiser",
            "domain",
        ]

        # not keeping 'userid' or 'IP' because thats just expected clcik per user
        #  'url', 'slotid', 'slotwidth', 'slotheight',
        #  'slotvisibility', 'slotformat', 'slotprice', 'keypage',
        #   'usertag'], # select the data features

    def calculate(self, data):
        """
        Calculates the expected cost per click
        sum(cost) / num_clicks
        """
        numerator = self.train.loc[self.train["click"] == 1, "payprice"].sum()
        denominator = self.train.loc[self.train["click"] == 1, "click"].sum()
        return numerator / denominator

    def getPossibleValues(self, feature):
        try:
            return sorted(self.data[feature].unique())
        except:
            print("Error, returning empty")
            print("feature didn't work:", feature)
            return []

    def calc_ecpc_given_feature(self, feature, value):
        numerator = self.data.loc[
            (self.data["click"] == 1) & (self.data[feature] == value), "payprice"
        ].sum()
        denominator = self.data.loc[
            (self.data["click"]) == 1 & (self.data[feature] == value), "click"
        ].sum()
        if denominator == 0 or numerator == 0:
            return 0
        return numerator / denominator

    def calc_per_feature(self):
        for feat in self.feature_list:
            expected_price_per_click_given_feature = []
            possible_values = self.getPossibleValues(feat)
            possible_values = list(
                map(
                    lambda x: (x, self.calc_ecpc_given_feature(feat, x)),
                    possible_values,
                )
            )
            self.plottable_items.append((feat, possible_values))

    def plot_ecpc_features(self):
        """
        plot the plottable items
        """
        for item in self.plottable_items:
            x_ticks = list(map(lambda x: x[0], item[1]))
            if len(x_ticks) > 0:
                plt.plot(range(len(x_ticks)), list(map(lambda x: x[1], item[1])))
                rotation = "horizontal"
                if type(x_ticks[0]) == str:
                    rotation = "vertical"
                plt.xticks(list(range(len(x_ticks))), x_ticks, rotation=rotation)
                plt.title(item[0])
                plt.show()


if __name__ == "__main__":
    ecpc = ECPC()
    ecpc.calc_per_feature()
    ecpc.plot_ecpc_features()
