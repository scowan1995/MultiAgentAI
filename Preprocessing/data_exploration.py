import matplotlib.pyplot as plt


def compute_cost():
    pass


def get_possible_values(single_set, feature):
    try:
        return sorted(single_set.data_features[feature].unique())
    except:
        print("Error, returning empty")
        print("feature didn't work:", feature)
        return []


def compute_ecpc_single_feature(single_set, feature, value):
    rows_to_select = (single_set.data_targets["click"] == 1) & \
                     (single_set.data_features[feature] == value)
    numerator = single_set.data_targets.copy().loc[rows_to_select, "payprice"].sum()

    denominator = single_set.data_targets.loc[rows_to_select, "click"].sum()

    result = 0
    if denominator != 0:
        result = numerator / denominator
    return result


def compute_ecpc_multiple_features(single_set, features_list=None):
    if features_list is None:
        features_list = [
            "weekday",
            "hour",
            "useragent",
            "region",
            "city",
            "adexchange",
            "creative",
            "advertiser",
            "domain",
        ]

    plottable_items = []
    for feature in features_list:
        possible_values = get_possible_values(single_set, feature)
        possible_values = list(
            map(
                lambda x: (x, compute_ecpc_single_feature(single_set, feature, x)),
                possible_values
            )
        )
        plottable_items.append((feature, possible_values))
    return plottable_items


def plot_ecpc_features(plottable_items):
    """
    plot the plottable items
    """
    for item in plottable_items:
        x_ticks = list(map(lambda x: x[0], item[1]))
        if len(x_ticks) > 0:
            plt.figure()
            plt.plot(range(len(x_ticks)), list(map(lambda x: x[1], item[1])))
            rotation = "horizontal"
            if type(x_ticks[0]) == str:
                rotation = "vertical"
            plt.xticks(list(range(len(x_ticks))), x_ticks, rotation=rotation)
            plt.title(item[0])
            plt.draw()
