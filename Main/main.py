import sys

import numpy as np
from Preprocessing.preprocessing import *
from Preprocessing.data_exploration import *
from Configs.configs import configs
from Model.logistic_regression import Logistic_Regression
from Bidders.constant_bidding_agent import ConstantBiddingAgent
from Bidders.random_bidding_agent import RandomBiddingAgent
from Rtb.rtb_ad_exchange import RtbAdExchange
# from Model.neural_network import

sys.path.append("../")


def single_agent_interact_with_rtb(bidder, print_results=False):
    for (_, features_row), (_, targets_row) in zip(sets['mock'].get_feature_iterator(),
                                                   sets['mock'].get_targets_iterator()):

        rtb.evaluate_known_auction(targets_row)

        # agent bids evaluating info received from RTB ad exchange and DMP
        if bidder.can_bid:
            bid_value = bidder.bid(ad_user_auction_info=features_row)
            rtb.receive_new_bid(bid_value)

        pay_price, click = rtb.report_win_notice()

        # agent receives win notice from RTB ad exchange (until his last bid => before finishing budget)
        if bidder.can_bid:
            bidder.read_win_notice(cost=pay_price, click=click)

    if print_results:
        print(f"Final budget = {bidder.get_current_budget()}. "
              f"Clicks obtained = {bidder.clicks_obtained}. "
              f"Click Through Rate = {bidder.get_current_click_through_rate()}. "
              f"Cost Per Click = {bidder.get_current_cost_per_click()}")

    result = np.array([bidder.get_current_budget(), bidder.clicks_obtained,
                       bidder.get_current_click_through_rate(), bidder.get_current_cost_per_click()])

    return result


def multiple_random_bidders_interact_with_rtb(bidder_agents):
    auction_counter = 0
    for (_, features_row), (_, targets_row) in zip(sets['mock'].get_feature_iterator(),
                                                   sets['mock'].get_targets_iterator()):

        rtb.evaluate_known_auction(targets_row)

        for current_bidder in bidder_agents:
            if current_bidder.can_bid:
                bid_value = current_bidder.bid(ad_user_auction_info=features_row)
                rtb.receive_new_bid(bid_value)

        pay_price, click = rtb.report_win_notice()
        auction_counter += 1

        for bidder_number, current_bidder in enumerate(bidder_agents):
            if current_bidder.can_bid:
                if current_bidder.read_win_notice(cost=pay_price, click=click):
                    print(f"The winner of auction {auction_counter} is bidder number {bidder_number}")


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
    bidder_budget = 6250 * 1000

    # SINGLE CONSTANT BIDDER_______________________________________________________
    if configs['constant_bidding']:
        # define bidder using 'train' set
        constant_bidder = ConstantBiddingAgent(training_set=sets['mock'],
                                               initial_budget=bidder_budget)
        single_agent_interact_with_rtb(constant_bidder, print_results=True)

    # SINGLE RANDOM BIDDER_________________________________________________________
    if configs['random_bidding']:
        # define bidder using 'train' set
        random_bidder = RandomBiddingAgent(training_set=sets['mock'],
                                           initial_budget=bidder_budget)
        # normal usage of the bidder
        single_agent_interact_with_rtb(random_bidder, print_results=True)

        # repeated usage of the bidder for its analysis
        random_bidder.perturbation = True  # The boundaries are perturbed with respect to training
        results = []
        all_boundaries = []
        total_iterations = 10
        for iteration in range(total_iterations):
            results.append(single_agent_interact_with_rtb(random_bidder))
            all_boundaries.append(random_bidder.get_boundaries())
        results_np = np.array(results)
        best_cpc_index = int(np.argmin(results_np[:, 3]))
        print(f"Random bidder, results after {total_iterations} iterations: Best values -> ",
              f"boundaries={all_boundaries[best_cpc_index]}; final budget={results_np[best_cpc_index, 0]}; ",
              f"clicks={results_np[best_cpc_index, 1]}; ctr={results_np[best_cpc_index, 2]}; ",
              f"cpc={results_np[best_cpc_index, 3]}. Cpc mean and std=[{np.mean(results_np[:, 3])},",
              f"{np.std(results_np[:, 3])}]. Ctr mean and std={np.mean(results_np[:, 2])}, {np.std(results_np[:, 2])}]")

    # MULTIPLE RANDOM BIDDERS______________________________________________________
    if configs['multiple_random_bidding']:
        total_bidders = 10
        bidders = []
        for i in range(total_bidders):
            random_bidder = RandomBiddingAgent(training_set=sets['mock'],
                                               initial_budget=bidder_budget)
            random_bidder.perturbation = True
            bidders.append(random_bidder)

        multiple_random_bidders_interact_with_rtb(bidders)

    if configs['logistic_regression']:
        logistic_regression = Logistic_Regression(sets['mock'].data_features, sets['mock'].data_targets)
        print(logistic_regression.score)

    plt.show()
