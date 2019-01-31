import sys
import os
sys.path.append("../../")


from Configs.configs import statics, configs


import csv
import pandas as pd
from sklearn.preprocessing import LabelEncoder




def load_data(data):
    """
    loads the data from csv to pandas
    :param data: statics['data']['mock'] path to data file
    :return: loaded pandas
    """

    ## load data and fill all Null values with 0.0
    data_pandas = pd.read_csv(data, na_values=['(NA)']).fillna(0)

    return data_pandas




def feature_target(data):
    """
    receives a dataframe and splits it up into features and targets
    :param data:
    :return:
    """

    targets = data['click'].astype('category')
    features = pd.DataFrame(data.pop('click'))

    return features, targets




# main ______________________________________________________________________________________

if __name__ == "__main__":


    ## TRAIN______________________________________________________________________________

    dataset = load_data(os.path.abspath(__file__ + "/../../") + statics['data']['mock'])
    column_names = list(dataset.columns.values)

    #print("total samples:", len(dataset))
    #print("column_names", column_names)

    train_features, train_targets = feature_target(dataset)







