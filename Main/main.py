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

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # used for 3D plot

sys.path.append("../")


def main():
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
        single_agent_interact_with_rtb(constant_bidder, rtb, sets, print_results=True)

    # SINGLE RANDOM BIDDER_________________________________________________________
    if configs['random_bidding']:
        # define bidder using 'train' set
        random_bidder = RandomBiddingAgent(training_set=sets['mock'],
                                           initial_budget=bidder_budget)
        # normal usage of the bidder
        single_agent_interact_with_rtb(random_bidder, rtb, sets, print_results=True)

        # repeated usage of the bidder for its analysis
        random_bidder.perturbation = True  # The boundaries are perturbed with respect to training
        results = []
        all_boundaries = []
        total_iterations = 10
        for iteration in range(total_iterations):
            results.append(single_agent_interact_with_rtb(random_bidder, rtb, sets))
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
        total_bidders = 100
        bidders = []
        for i in range(total_bidders):
            random_bidder = RandomBiddingAgent(training_set=sets['train'],
                                               initial_budget=bidder_budget)
            # If bidders share the same dataset, manually perturb their "training boundaries" once
            lower = np.random.uniform(random_bidder.get_boundaries()[0], random_bidder.get_mean())
            upper = np.random.uniform(random_bidder.get_mean(), random_bidder.get_boundaries()[1])
            random_bidder.set_boundaries(lower, upper)

            # random_bidder.perturbation = True  # further boundary perturbation (at every bid)
            bidders.append(random_bidder)

        multiple_random_bidders_interact_with_rtb(bidders, rtb, sets)
        evaluate_multiple_random_bidders_performance(bidders)

    if configs['logistic_regression']:
        logistic_regression = Logistic_Regression(sets['mock'].data_features, sets['mock'].data_targets)
        print(logistic_regression.score)


def single_agent_interact_with_rtb(bidder, rtb, sets, print_results=False):
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


def multiple_random_bidders_interact_with_rtb(bidder_agents, rtb, sets):
    auction_counter = 0
    for (_, features_row), (_, targets_row) in zip(sets['val'].get_feature_iterator(),
                                                   sets['val'].get_targets_iterator()):

        rtb.evaluate_known_auction(targets_row)

        for current_bidder in bidder_agents:
            if current_bidder.can_bid:
                bid_value = current_bidder.bid(ad_user_auction_info=features_row)
                rtb.receive_new_bid(bid_value)

        pay_price, click = rtb.report_win_notice()
        auction_counter += 1

        for bidder_number, current_bidder in enumerate(bidder_agents):
            if current_bidder.can_bid:
                current_bidder.read_win_notice(cost=pay_price, click=click)


def evaluate_multiple_random_bidders_performance(multiple_bidders):
    lower_boundaries = []
    upper_boundaries = []
    ctrs = []
    cpcs = []
    clicks = []
    paied = []
    for bidder in multiple_bidders:
        low, high = bidder.get_boundaries()
        lower_boundaries.append(low)
        upper_boundaries.append(high)
        ctrs.append(bidder.get_current_click_through_rate())
        clicks.append(bidder.clicks_obtained)
        paied.append(bidder.get_total_paied())
        cpcs.append(bidder.get_current_cost_per_click())

    clicks_np = np.array(clicks)
    oredered_indeces = np.argsort(clicks_np)
    lower_boundaries_np = np.array(lower_boundaries)
    upper_boundaries_np = np.array(upper_boundaries)
    ctrs_np = np.array(ctrs)
    paied_np = np.array(paied)
    cpcs_np = np.array(cpcs)

    print(f"Results with {len(multiple_bidders)} bidders: ")
    print(f"best ctr mean {np.mean(ctrs_np[oredered_indeces[-10:]])} and "
          f"std {np.std(ctrs_np[oredered_indeces[-10:]])}")
    print(f"lower boundaries mean {np.mean(lower_boundaries_np[oredered_indeces[-10:]])} and "
          f"std {np.std(lower_boundaries_np[oredered_indeces[-10:]])}")
    print(f"upper boundaries mean {np.mean(upper_boundaries_np[oredered_indeces[-10:]])} and "
          f"std {np.std(upper_boundaries_np[oredered_indeces[-10:]])}")
    print(f"clicks mean {np.mean(clicks_np[oredered_indeces[-10:]])} and "
          f"std {np.std(clicks_np[oredered_indeces[-10:]])}")
    print(f"paied mean {np.mean(paied_np[oredered_indeces[-10:]])} and "
          f"std {np.std(paied_np[oredered_indeces[-10:]])}")
    print(f"cpc mean {np.mean(cpcs_np[oredered_indeces[-10:]])} and "
          f"std {np.std(cpcs_np[oredered_indeces[-10:]])}")

    ranked_plot(lower_boundaries_np, upper_boundaries_np, ctrs_np, clicks_np )


def ranked_plot(x, y, z, t):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    min_x = min(x)
    max_x = max(x)
    min_x -= (max_x - min_x) / 10
    max_x += (max_x - min_x) / 10

    min_y = min(y)
    max_y = max(y)
    min_y -= (max_y - min_y) / 10
    max_y += (max_y - min_y) / 10

    min_z = min(z)
    max_z = max(z)
    min_z -= (max_z - min_z) / 10
    max_z += (max_z - min_z) / 10

    ax.scatter(x, z, marker='+', color='darkturquoise', zdir='y', zs=max_y)
    ax.scatter(y, z, marker='+', color='cadetblue', zdir='x', zs=min_x)
    # ax.plot(x, y, 'k+', zdir='z', zs=min_z)

    # points with higher number of clicks
    oredered_indeces = np.argsort(t)
    cmap = matplotlib.cm.get_cmap('coolwarm')
    normalize = matplotlib.colors.Normalize(vmin=min(z[oredered_indeces[-10:]]),
                                            vmax=max(z[oredered_indeces[-10:]]))
    colors = [cmap(normalize(value)) for value in z[oredered_indeces[-10:]]]

    ax.scatter(x[oredered_indeces[-10:]],
               y[oredered_indeces[-10:]],
               z[oredered_indeces[-10:]], color=colors, s=100, edgecolors='none')

    ax.scatter(x, y, z, color='midnightblue')

    ax.set_xlim([min_x, max_x])
    ax.set_ylim([min_y, max_y])
    ax.set_zlim([min_z, max_z])

    ax.set_xlabel("lower bound")
    ax.set_ylabel("upper bound")
    ax.set_zlabel("CTR", rotation=270, labelpad=5)

    plt.draw()
    plt.savefig("../Plots/multiple_rand2.svg", transparent=True, format='svg', frameon=False)


if __name__ == "__main__":
    main()
    plt.show()
