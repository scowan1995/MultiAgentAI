import sklearn.model_selection as ms
from .single_set import SingleSet
from Model.data_exploration import DataExploration
from Configs.configs import statics, configs


def merge_sets(set_1, set_2, use_numerical_labels=True):
    new_data = set_1.data.append(set_2.data, ignore_index=True, sort=True)
    new_set = SingleSet(data=new_data)
    if use_numerical_labels:
        new_set.numericalize_labels()
    new_set.split_in_feature_and_target()
    return new_set


def split_sets(initial_set, size_percentage_first_set=50, use_numerical_labels=True):
    data_1, data_2 = ms.train_test_split(initial_set.data, test_size=(100 - size_percentage_first_set) / 100)
    set_1 = SingleSet(data=data_1)
    set_2 = SingleSet(data=data_2)
    if use_numerical_labels:
        set_1.numericalize_labels()
        set_2.numericalize_labels()
    set_1.split_in_feature_and_target()
    set_2.split_in_feature_and_target()
    return set_1, set_2
