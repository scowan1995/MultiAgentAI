import sys
import os

sys.path.append("../../")

from Configs.configs import statics, configs
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle


# Class Data Loader and Preprocessing

class LoadPreprocess(object):
    def __init__(self, relative_path, use_numerical_labels=True):
        self.data = self.load_data(relative_path)
        self.data_features = None
        self.data_targets = None
        self.label_to_numerical_mapping = None

        self.split_in_feature_and_target()
        if use_numerical_labels:
            self.numericalize_labels()

        """
        if configs["combine_train_val"]:

            # TRAIN AND VAL ________________________________________________________________

            if configs["load_pickle"]:

                try:
                    self.train_val_features_num = pd.read_pickle(
                        os.path.abspath(__file__ + "/../../Data/train_val_features_num")
                    )
                    self.train_val_features = pd.read_pickle(
                        os.path.abspath(__file__ + "/../../Data/train_val_features")
                    )
                    self.train_val_targets = pd.read_pickle(
                        os.path.abspath(__file__ + "/../../Data/train_val_targets")
                    )
                    with open(
                        os.path.abspath(__file__ + "/../../Data/train_val_mapping"),
                        "rb",
                    ) as file:
                        self.train_val_mapping = pickle.load(file)

                except:
                    print(
                        "load_pickle should be set to False because the data has not been loaded yet"
                    )

            else:
                self.train_val = pd.concat(
                    [self.train_data, self.val_data], axis=0, sort=False
                )

                self.train_val_features, self.train_val_targets = self.split_in_feature_and_target(
                    self.train_val
                )
                self.train_val_features_num, self.train_val_mapping = self.numericalize_labels(
                    self.train_val_features
                )

                self.train_val_features_num.to_pickle(
                    os.path.abspath(__file__ + "/../../Data/train_val_features_num")
                )
                self.train_val_features.to_pickle(
                    os.path.abspath(__file__ + "/../../Data/train_val_features")
                )
                self.train_val_targets.to_pickle(
                    os.path.abspath(__file__ + "/../../Data/train_val_targets")
                )
                with open(
                    os.path.abspath(__file__ + "/../../Data/train_val_mapping"), "wb"
                ) as file:
                    pickle.dump(
                        self.train_val_mapping, file, protocol=pickle.HIGHEST_PROTOCOL
                    )

                print("data has been stored as pickle")
        """
        print("-- data loaded --")

    @staticmethod
    def load_data(dataset_rel_path):
        """
        loads the data from csv to pandas and fill all Null values with 0.0
        :param dataset_rel_path: statics['data']['mock'] path to data file
        :return: loaded pandas
        """
        path = os.path.abspath(__file__ + "/../../") + dataset_rel_path
        data_pandas = pd.read_csv(path, na_values=["Na", "null"]).fillna(0)
        return data_pandas

    def split_in_feature_and_target(self):
        """
        Splits data dataframe into features and targets

        """
        self.data_targets = self.data[configs["target"]]  # .astype('category')

        # drop target
        features = pd.DataFrame(self.data)
        # drop features and targets
        for f in features.columns.values:
            if f not in configs["features"]:
                features.pop(f)
        self.data_features = features

    def numericalize_labels(self):
        """
        numericalizes categorical columns in pandas dataframe
        :param data: pandas dataframe
        :return: numericalized dataframe and mapping between numbers and previous categorical values
        """
        le = LabelEncoder()
        le_mapping = dict()
        data = self.data
        for col in data.columns.values:
            print("numericalizing", col)
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

        # Importing MinMaxScaler and initializing it
        from sklearn.preprocessing import MinMaxScaler

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


# main ______________________________________________________________________________________

if __name__ == "__main__":
    print("preprocessing")

