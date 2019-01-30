import sys
import os
sys.path.append("../../")


from Configs.configs import statics, configs


import csv
import pandas as pd



# mock ________________________________________________________________________________________

def load_data(data):
    """

    :param data: statics['data']['mock']
    :return: loaded pandas
    """
    data_pandas = pd.read_csv(data)

    return data_pandas




# main ______________________________________________________________________________________

if __name__ == "__main__":

    mock = load_data(os.path.abspath(__file__ + "/../../") + statics['data']['mock'])
    #train = load_data(os.path.abspath(__file__ + "/../../") + statics['data']['train'])
    #val = load_data(os.path.abspath(__file__ + "/../../") + statics['data']['val'])
    #test = load_data(os.path.abspath(__file__ + "/../../") + statics['data']['test'])

    print(mock)