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


def load_all_datasets_and_explore_them(available_datasets):
    loaded_datasets = {}
    for current_dataset_name in available_datasets:
        if configs[current_dataset_name]:
            loaded_datasets[current_dataset_name] = SingleSet(statics['data'][current_dataset_name],
                                                              use_numerical_labels=True)
            if configs['data_exploration']:
                data_exploration = DataExploration(loaded_datasets[current_dataset_name])
    return loaded_datasets

if __name__ == "__main__":


    ## DATA_________________________________________________________________________

    datasets_names = ['mock', 'train', 'val', 'test']
    datasets = load_all_datasets_and_explore_them(datasets_names)


    ## MODEL________________________________________________________________________

    if configs['logistic_regression']:

        logistic_regression = Logistic_Regression(datasets['mock'].data_features, datasets['mock'].data_targets)
        print(logistic_regression.score)















