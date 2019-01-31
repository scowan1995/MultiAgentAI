
import sys
import os

sys.path.append("../../")


from Configs.configs import statics, configs
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle





def load_data(data):
    """
    loads the data from csv to pandas
    :param data: statics['data']['mock'] path to data file
    :return: loaded pandas
    """

    ## load data and fill all Null values with 0.0
    data_pandas = pd.read_csv(data, na_values=['Na', 'null']).fillna(0)

    return data_pandas






def feature_target(data):
    """
    receives a dataframe and splits it up into features and targets
    :param data: pandas dataframe of loaded csv
    :return: features, targets of whole dataframe
    """

    targets = data[configs['target']].astype('category')

    # drop target
    features = pd.DataFrame(data)

    #drop features and targets
    for f in features.columns.values:
        if f not in configs['features']:
            features.pop(f)


    return features, targets







def numericalize(data):
    """
    numericalizes categorical columns in pandas dataframe
    :param data: pandas dataframe
    :return: numericalized dataframe and mapping between numbers and previous categorical values
    """
    le = LabelEncoder()
    le_mapping = dict()

    for col in data.columns.values:

        print("numericalizing", col)

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






def scale(data):
    """
    scale numerical data between 0 and 1
    :param data: pandas dataframe with numericalized values
    :return: returns scaled pandas dataframe with numericalized values
    """

    # Importing MinMaxScaler and initializing it
    from sklearn.preprocessing import MinMaxScaler
    min_max = MinMaxScaler()
    # Scaling down both train and test data set
    scaled_data = min_max.fit_transform(data[['weekday', 'hour', 'bidid', 'userid', 'useragent', 'IP', 'region', 'city', 'adexchange', 'domain', 'url', 'urlid', 'slotid', 'slotwidth', 'slotheight', 'slotvisibility', 'slotformat', 'slotprice', 'creative', 'bidprice', 'payprice', 'keypage', 'advertiser', 'usertag']])

    return scaled_data





## Class Data Loader and Preprocessing

class Load_Preprocess(object):

    def __init__(self):


        ## MOCK ______________________________________________________________________________

        if configs['mock'] == True:

            dataset = load_data(os.path.abspath(__file__ + "/../../") + statics['data']['mock'])
            column_names = list(dataset.columns.values)

            # print("total samples:", len(dataset))
            # print("column_names", column_names)

            self.features, self.targets = feature_target(dataset)
            self.features_num, self.mapping = numericalize(self.features)


        else:

            ## TRAIN AND VAL ________________________________________________________________

            if configs['load_pickle']:

                try:

                    self.features_num = pd.read_pickle(os.path.abspath(__file__ + "/../../Data/features_num"))
                    self.features = pd.read_pickle(os.path.abspath(__file__ + "/../../Data/features"))
                    self.targets = pd.read_pickle(os.path.abspath(__file__ + "/../../Data/targets"))

                    with open(os.path.abspath(__file__ + "/../../Data/mapping"), 'rb') as file:
                        self.mapping = pickle.load(file)

                except:

                    print('load_pickle should be set to False because the data has not been loaded yet')

            else:

                train = load_data(os.path.abspath(__file__ + "/../../") + statics['data']['train'])
                val = load_data(os.path.abspath(__file__ + "/../../") + statics['data']['val'])

                train_val = pd.concat([train, val], axis=0, sort=False)

                self.features, self.targets = feature_target(train_val)
                self.features_num, self.mapping = numericalize(self.features)

                self.features_num.to_pickle(os.path.abspath(__file__ + "/../../Data/features_num"))
                self.features.to_pickle(os.path.abspath(__file__ + "/../../Data/features"))
                self.targets.to_pickle(os.path.abspath(__file__ + "/../../Data/targets"))

                with open(os.path.abspath(__file__ + "/../../Data/mapping"), 'wb') as file:
                    pickle.dump(self.mapping, file, protocol=pickle.HIGHEST_PROTOCOL)

                print('data has been stored as pickle')

        print("-- data loaded --")


# main ______________________________________________________________________________________

if __name__ == "__main__":

    print("preprocessing")


