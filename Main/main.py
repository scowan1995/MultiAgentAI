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
    bidder_budget = 1000
    if configs['constant_bidding']:
        # define bidder using 'train' set
        constant_bidder = ConstantBiddingAgent(training_set=sets['mock'].data_features,
                                               initial_budget=bidder_budget)

        # iterate over 'validation set'
        for (_, features_row), (_, targets_row) in zip(sets['mock'].get_feature_iterator(),
                                                       sets['mock'].get_targets_iterator()):

            # agent bids evaluating info received from RTB ad exchange and DMP
            if constant_bidder.can_bid:
                constant_bidder.bid(ad_user_auction_info=features_row)

                # agent receives win notice from RTB ad exchange
                click = bool(targets_row["click"])
                pay_price = targets_row["payprice"]
                constant_bidder.read_win_notice(cost=pay_price, click=click)

            else:
                print("Budged finished!")
                break

        print(f"Final budget = {constant_bidder.get_current_budget()}")

    if configs['logistic_regression']:
        logistic_regression = Logistic_Regression(sets['mock'].data_features, sets['mock'].data_targets)
        print(logistic_regression.score)

    plt.show()
