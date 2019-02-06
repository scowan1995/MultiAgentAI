import os
import pandas as pd
import sklearn.model_selection as ms


class DataLoader:
    """
    Takes care of loading the data from the CSV, splitting to training,
    validation, test
    """

    def __init__(self, use_pickle=True, load_small=False):
        """
        params:
            use_pickle:
            determines whether to use the pickled data or if it doesn't exist
            a new pickle file will be saved

            load_small:
            determines whether to speed things up by loading a very small
            amount of data to allow us to test things quickly
        """
        self.pickle_name = "train_test.pickle"
        self.use_pickle = use_pickle
        self.load_small = load_small
        self.train_name = "train.csv"
        self.train_small_name = "train_small.csv"
        self.test_name = "test.csv"
        self.test_small_name = "test_small.csv"
        if load_small:
            self.train_path = self.train_small_name
        else:
            self.train_path = self.train_name

    def get_train_val(self, validation_size=0.1):
        """
        loads and seperates the training data into train and validation set
        """
        assert validation_size < 1
        assert validation_size >= 0

        if self.use_pickle and os.path.isfile("./Data/" + self.pickle_name):
            pass
        else:
            data = pd.read_csv(
                "./Data/" + self.train_path, na_values=["Na", "null"]
            ).fillna(0.0)
            train, test = ms.train_test_split(data, test_size=validation_size)
            return train, test

    def convert_to_features_labels(self, data):
        """
        Splits the data into the features and the labels
        """
        clicks_ind = data.columns.tolist().index("click")
        payPrice_ind = data.columns.tolist().index("payprice")

if __name__ == "__main__":
    dl = DataLoader(load_small=True)
    train, test = dl.get_train_val()
    print("train:", train.shape)
    print("test:", test.shape)
    dl.convert_to_features_labels(train)
