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

    data = os.path.abspath(__file__ + "/../../") + statics['data']['mock']
    data_pandas = load_data(data)

    print(data_pandas)