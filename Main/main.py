import os
import sys
import numpy as np
import pandas as pd
sys.path.append("../")

from Configs.configs import statics, configs
from Preprocessing.preprocessing import Load_Preprocess
from Model.logistic_regression import Logistic_Regression
#from Model.neural_network import


if __name__ == "__main__":


    ## DATA_________________________________________________________________________

    # data object contains "features", "targets", "features_num" and "mapping"
    data = Load_Preprocess()


    ## MODEL________________________________________________________________________

    if configs['logistic_regression']:

        logistic_regression = Logistic_Regression(data.features_num, data.targets)
        print(logistic_regression.score)













