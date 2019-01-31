import sys
import os
sys.path.append("../../")

from Configs.configs import statics, configs
from Loading.dataload import load_data, feature_target
from Preprocessing.preprocessing import scale, numericalize


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def train_logRegression(train_features, train_targets):

    logisticRegr = LogisticRegression()
    logisticRegr.fit(train_features, train_targets)

    return logisticRegr



def train_val_split(features, labels):

    train_features, val_features, train_labels, val_labels = train_test_split(features, labels, test_size=configs['val_ratio'], random_state=0)

    return train_features, val_features, train_labels, val_labels



def validate(logisticRegr, features, labels):

    score = logisticRegr.score(features, labels)

    return score



## Class Logistic Regression

class Logistic_Regression(object):

    def __init__(self, features, targets):

        self.features = features
        self.targets = targets

        train_features, val_features, train_labels, val_labels = train_val_split(features, targets)
        self.logisticRegr = train_logRegression(train_features, train_labels)

        self.score = validate(self.logisticRegr, val_features, val_labels)



# main ______________________________________________________________________________________

if __name__ == "__main__":


    dataset = load_data(os.path.abspath(__file__ + "/../../") + statics['data']['mock'])
    column_names = list(dataset.columns.values)

    # print("total samples:", len(dataset))
    # print("column_names", column_names)

    features, targets = feature_target(dataset)
    features_num, mapping = numericalize(features)

    train_features, val_features, train_labels, val_labels = train_val_split(features, targets)
    logisticRegr = train_logRegression(train_features, train_labels)

    score = validate(logisticRegr, val_features, val_labels)

    print(score)

