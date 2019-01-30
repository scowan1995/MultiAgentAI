import os
import sys
sys.path.append("../")

from Configs.configs import statics, configs
from Loading.dataload import load_data


if __name__ == "__main__":

    mock = load_data(os.path.abspath(__file__ + "/../../") + statics['data']['mock'])