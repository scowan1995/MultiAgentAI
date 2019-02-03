import sys
import os
sys.path.append("../../")

from Configs.configs import statics, configs
from Preprocessing.preprocessing import scale, numericalize
from Configs.configs import statics, configs
from Preprocessing.preprocessing import Load_Preprocess


import numpy as np
import pandas as pd




class Data_Exploration(object):

    def __init__(self, data):

        self.data = data


# main ______________________________________________________________________________________

if __name__ == "__main__":

    ## DATA_________________________________________________________________________

    # data object contains "features", "targets", "features_num" and "mapping"
    data = Load_Preprocess()


    # DATA EXPLORATION________________________________________________________________

    data_exploration = Data_Exploration(data.features_num, data.targets)
