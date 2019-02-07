import os
import sys
import numpy as np
import pandas as pd
sys.path.append("../")

from Configs.configs import statics, configs
from Preprocessing.preprocessing import SingleSet
from Model.data_exploration import Data_Exploration
from Model.logistic_regression import Logistic_Regression
#from Model.neural_network import


def load_all_datasets_and_explore_them(sets_info):
    loaded_sets = {}
    for current_set_name, loading_flag in sets_info.items():
        if loading_flag:
            loaded_sets[current_set_name] = SingleSet(statics['data'][current_set_name],
                                                      use_numerical_labels=True)
            if configs['data_exploration']:
                data_exploration = DataExploration(loaded_sets[current_set_name])
    return loaded_sets


if __name__ == "__main__":

    # DATA_________________________________________________________________________
    sets_information = configs['sets']
    sets = load_all_datasets_and_explore_them(sets_information)

    # MODEL________________________________________________________________________
    if configs['logistic_regression']:
        logistic_regression = Logistic_Regression(sets['mock'].data_features, sets['mock'].data_targets)
        print(logistic_regression.score)
