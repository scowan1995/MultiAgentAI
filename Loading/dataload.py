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




def numericalize(data):
    """
    numericalizes categorical columns in pandas dataframe
    :param data: pandas dataframe
    :return: numericalized dataframe and mapping between numbers and previous categorical values
    """
    le = LabelEncoder()
    le_mapping = dict()

    for col in data.columns.values:

        # Encoding only categorical variables
        if data[col].dtypes == 'object':
            # Using whole data to form an exhaustive list of levels

            categoricals = data[col].append(data[col])
            le.fit(categoricals.values.astype(str))
            ## safe mapped data
            data[col] = le.transform(data[col].astype(str))
            ## safe mapping
            le_mapping[col] = dict(zip(le.classes_, le.transform(le.classes_)))

    return data, le_mapping




def feature_label(data):
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

    train_features, train_targets = feature_label(dataset)
    train_features_num, train_mapping = numericalize(train_features)


    ## VAL______________________________________________________________________________

    dataset = load_data(os.path.abspath(__file__ + "/../../") + statics['data']['val'])
    column_names = list(dataset.columns.values)

    # print("total samples:", len(dataset))
    # print("column_names", column_names)

    val_features, val_targets = feature_label(dataset)
    val_features_num, val_mapping = numericalize(train_features)









