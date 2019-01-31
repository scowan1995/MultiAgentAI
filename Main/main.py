import os
import sys
import pandas as pd
sys.path.append("../")

from Configs.configs import statics, configs
from Loading.dataload import load_data, feature_target
from Preprocessing.preprocessing import scale, numericalize


if __name__ == "__main__":

    ## MOCK ______________________________________________________________________________

    if configs['mock'] == True:

        dataset = load_data(os.path.abspath(__file__ + "/../../") + statics['data']['mock'])
        column_names = list(dataset.columns.values)

        # print("total samples:", len(dataset))
        # print("column_names", column_names)

        features, targets = feature_target(dataset)
        features_num, mapping = numericalize(features)

    else:

    ## TRAIN AND VAL________________________________________________________________

        train = load_data(os.path.abspath(__file__ + "/../../") + statics['data']['train'])
        val = load_data(os.path.abspath(__file__ + "/../../") + statics['data']['val'])

        train_val = pd.concat([train, val], axis=0, sort=False)

        features, targets = feature_target(train_val)
        features_num, mapping = numericalize(features)













