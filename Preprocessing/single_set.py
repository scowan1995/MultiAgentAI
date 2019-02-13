import sys
import os
import pandas as pd
import pickle
from Configs.configs import statics, configs
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

sys.path.append("../../")


# Class Data Loader and Preprocessing
class SingleSet(object):
    def __init__(self, relative_path=None, data=None, use_numerical_labels=True):
        self.data = data
        self.data_features = data
        self.data_targets = None
        self.label_to_numerical_mapping = None

        if relative_path is not None:
            try:
                self.load_pickle(relative_path)
            except FileNotFoundError:
                self.load_data(relative_path)
                self.data_features = self.data

                if use_numerical_labels:
                    # it has to be executed before the splitting
                    self.numericalize_labels()

                self.split_in_feature_and_target()
                self.save_pickle(relative_path)

            print("-- data loaded --")

    def load_data(self, dataset_rel_path):
        """
        loads the data from csv to pandas and fill all Null values with 0.0
        :param dataset_rel_path: statics['data']['mock'] path to data file
        """
        path = os.path.abspath(__file__ + "/../../") + dataset_rel_path
        data_pandas = pd.read_csv(path, na_values=["Na", "null"]).fillna(0)
        self.data = data_pandas

    def load_pickle(self, relative_path):
        absolute_path = os.path.abspath(__file__ + "/../../" + relative_path[:-4] + "_pickle")
        with open(absolute_path, 'rb') as file_handler:
            loaded_object = pickle.load(file_handler)
        self.data = loaded_object.data
        self.data_features = loaded_object.data_features
        self.data_targets = loaded_object.data_targets
        self.label_to_numerical_mapping = loaded_object.label_to_numerical_mapping

    def save_pickle(self, relative_path):
        absolute_path = os.path.abspath(__file__ + "/../../" + relative_path[:-4] + "_pickle")
        with open(absolute_path, 'wb') as file_handler:
            pickle.dump(self, file_handler, protocol=pickle.HIGHEST_PROTOCOL)
        print("saved pickle")

    def split_in_feature_and_target(self):
        """
        Splits data dataframe into features and targets

        """
        if len(configs["target"]) == 1:
            self.data_targets = self.data_features[configs["target"]]
        else:
            targets = self.data_features.copy()
            # drop features and targets
            for t in targets.columns.values:
                if t not in configs["target"]:
                    targets.pop(t)
            self.data_targets = targets

        # drop target
        features = self.data_features.copy()
        # drop features and targets
        for f in features.columns.values:
            if f not in configs["features"]:
                features.pop(f)
        self.data_features = features

    def get_feature_iterator(self):
        try:
            return self.data_features.iterrows()
        except AttributeError:
            return self.data_features.iteritems()

    def get_targets_iterator(self):
        try:
            return self.data_targets.iterrows()
        except AttributeError:
            return self.data_targets.iteritems()

    def numericalize_labels(self):
        """
        numericalizes categorical columns in pandas dataframe
        """
        le = LabelEncoder()
        le_mapping = dict()
        data = self.data.copy()
        for col in data.columns.values:
            #print("numericalizing", col)
            # Encoding only categorical variables
            if data[col].dtypes == "object":
                # Using whole data to form an exhaustive list of levels
                categoricals = data[col].append(data[col])
                le.fit(categoricals.values.astype(str))
                # safe mapped data
                data[col] = le.transform(data[col].astype(str))
                # safe mapping
                le_mapping[col] = dict(zip(le.classes_, le.transform(le.classes_)))
        self.data_features = data
        self.label_to_numerical_mapping = le_mapping

    def scale(self):
        """
        scale numerical data between 0 and 1
        :return: returns scaled pandas dataframe with numericalized values
        """
        min_max = MinMaxScaler()
        # Scaling down both train and test data set
        scaled_data = min_max.fit_transform(
            self.data_features[
                [
                    "weekday",
                    "hour",
                    "bidid",
                    "userid",
                    "useragent",
                    "IP",
                    "region",
                    "city",
                    "adexchange",
                    "domain",
                    "url",
                    "urlid",
                    "slotid",
                    "slotwidth",
                    "slotheight",
                    "slotvisibility",
                    "slotformat",
                    "slotprice",
                    "creative",
                    "bidprice",
                    "payprice",
                    "keypage",
                    "advertiser",
                    "usertag",
                ]
            ]
        )

        return scaled_data
