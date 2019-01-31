import sys
import os
sys.path.append("../../")

from sklearn.linear_model import LogisticRegression
from Configs.configs import statics, configs


def train_logRegression(train_features, train_targets):

    logisticRegr = LogisticRegression()
    logisticRegr.fit(train_features, train_targets)

    return logisticRegr





# main ______________________________________________________________________________________

if __name__ == "__main__":

    print("logRegression")