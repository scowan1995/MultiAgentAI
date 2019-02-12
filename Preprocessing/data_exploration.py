import sys
import os
sys.path.append("../../")

from Configs.configs import statics, configs
from Configs.configs import statics, configs
from Preprocessing.preprocessing import SingleSet


import numpy as np
import pandas as pd


class DataExploration(object):

    def __init__(self, data):
        self._data = data

    def compute_cost(self):
        pass

    def compute_expected_cost_per_click(self):
        pass

