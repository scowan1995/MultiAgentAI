import numpy as np
import sklearn.model_selection as ms
from sklearn.preprocessing import MinMaxScaler
from .single_set import SingleSet
from Configs.configs import statics


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


def load_all_datasets(sets_info):
    loaded_sets = {}
    for current_set_name, loading_flag in sets_info.items():
        if loading_flag:
            loaded_sets[current_set_name] = SingleSet(relative_path=statics['data'][current_set_name],
                                                      use_numerical_labels=True)
    return loaded_sets


def scale_all_sets_features(sets):
    grouped = []
    for _, current_set in sets.items():
        set_features_np = np.asarray(current_set.data_features.values)
        grouped.append(set_features_np)
    grouped_np = np.vstack(grouped)

    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    feature_scaler.fit(grouped_np)

    for _, current_set in sets.items():
        current_set.data_features_scaled_np = feature_scaler.transform(current_set.data_features.values)
