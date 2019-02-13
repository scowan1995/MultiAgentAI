import sys

from Preprocessing.preprocessing import *
from Preprocessing.data_exploration import *
from Configs.configs import configs
from Model.logistic_regression import Logistic_Regression
from Bidders.constant_bidding_agent import ConstantBiddingAgent
# from Model.neural_network import

sys.path.append("../")

if __name__ == "__main__":

    # DATA_________________________________________________________________________
    sets_information = configs['sets']
    sets = load_all_datasets(sets_information)
    # s1, s2 = split_sets(sets['mock'])
    # s3 = merge_sets(sets['mock'], sets['mock'])

    ecpcs_plot_data = compute_ecpc_multiple_features(sets['mock'])
    plot_ecpc_features(ecpcs_plot_data)

    # MODEL________________________________________________________________________

    if configs['constant_bidding']:
        constant_bidder = ConstantBiddingAgent(sets['mock'].data_features, 10)
        constant_bidder.bid()
        print(constant_bidder._current_budget)


    if configs['logistic_regression']:
        logistic_regression = Logistic_Regression(sets['mock'].data_features, sets['mock'].data_targets)
        print(logistic_regression.score)

    plt.show()
