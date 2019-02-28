import sys

from Preprocessing.preprocessing import *
from Preprocessing.data_exploration import *
from Configs.configs import configs
from Model.logistic_regression import Logistic_Regression
from Bidders.constant_bidding_agent import ConstantBiddingAgent
from Rtb.rtb_ad_exchange import RtbAdExchange
# from Model.neural_network import

sys.path.append("../")

if __name__ == "__main__":

    # DATA_________________________________________________________________________
    sets_information = configs['sets']
    sets = load_all_datasets(sets_information)
    # s1, s2 = split_sets(sets['mock'])
    # s3 = merge_sets(sets['mock'], sets['mock'])

    # ecpcs_plot_data = compute_ecpc_multiple_features(sets['mock'])
    # plot_ecpc_features(ecpcs_plot_data)

    # MODEL________________________________________________________________________
    rtb = RtbAdExchange()
    bidder_budget = 6250

    if configs['constant_bidding']:
        # define bidder using 'train' set
        constant_bidder = ConstantBiddingAgent(training_set=sets['mock'],
                                               initial_budget=bidder_budget)

        # iterate over 'validation set'
        for (_, features_row), (_, targets_row) in zip(sets['mock'].get_feature_iterator(),
                                                       sets['mock'].get_targets_iterator()):

            rtb.evaluate_known_auction(targets_row)

            # agent bids evaluating info received from RTB ad exchange and DMP
            if constant_bidder.can_bid:
                bid_value = constant_bidder.bid(ad_user_auction_info=features_row)
                rtb.receive_new_bid(bid_value)

            pay_price, click = rtb.report_win_notice()

            # agent receives win notice from RTB ad exchange (until his last bid => before finishing budget)
            if constant_bidder.can_bid:
                constant_bidder.read_win_notice(cost=pay_price, click=click)

        print(f"CONSTANT BIDDER.  Final budget = {constant_bidder.get_current_budget()}. "
              f"Clicks obtained = {constant_bidder.clicks_obtained}. "
              f"Click Through Rate = {constant_bidder.get_current_click_through_rate()}. "
              f"Cost Per Click = {constant_bidder.get_current_cost_per_click()}")

    if configs['logistic_regression']:
        logistic_regression = Logistic_Regression(sets['mock'].data_features, sets['mock'].data_targets)
        print(logistic_regression.score)

    plt.show()
