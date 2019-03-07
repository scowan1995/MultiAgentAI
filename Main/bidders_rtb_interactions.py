import datetime

from .plot_utils import *


def single_agent_interact_with_rtb(bidder, rtb, sets, print_results=False):
    counter = 0
    for (_, features_row), (_, targets_row) in zip(sets['val'].get_feature_iterator(),
                                                   sets['val'].get_targets_iterator()):

        rtb.evaluate_known_auction(targets_row)

        # agent bids evaluating info received from RTB ad exchange and DMP
        if bidder.can_bid:
            bid_value = bidder.bid(ad_user_auction_info=features_row)
            rtb.receive_new_bid(bid_value)

        pay_price, click = rtb.report_win_notice()

        # agent receives win notice from RTB ad exchange (until his last bid => before finishing budget)
        if bidder.can_bid:
            bidder.read_win_notice(cost=pay_price, click=click)

            if counter % 1000 == 0:
                print(f"Iteration n {counter}. Bids won = {bidder.get_bids_won()}. "
                      f"Clicks = {bidder.clicks_obtained}. Budget = {bidder.get_current_budget()}")
        counter += 1

    if print_results:
        print(f"Final budget = {bidder.get_current_budget()}. "
              f"Clicks obtained = {bidder.clicks_obtained}. "
              f"Click Through Rate = {bidder.get_current_click_through_rate()}. "
              f"Cost Per Click = {bidder.get_current_cost_per_click()}")

    result = np.array([bidder.get_current_budget(), bidder.clicks_obtained,
                       bidder.get_current_click_through_rate(), bidder.get_current_cost_per_click()])

    return result


def multiple_random_bidders_interact_with_rtb(bidder_agents, rtb, sets):
    for (_, features_row), (_, targets_row) in zip(sets['val'].get_feature_iterator(),
                                                   sets['val'].get_targets_iterator()):

        rtb.evaluate_known_auction(targets_row)

        for current_bidder in bidder_agents:
            if current_bidder.can_bid:
                bid_value = current_bidder.bid(ad_user_auction_info=features_row)
                rtb.receive_new_bid(bid_value)

        pay_price, click = rtb.report_win_notice()

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


def multiagent_bidders_interact_with_rtb_to_generate_new_set(bidder_agents, rtb, sets):
    counter = 0
    for (_, features_row), (_, targets_row) in zip(sets['train'].get_feature_iterator(),
                                                   sets['train'].get_targets_iterator()):

        rtb.evaluate_known_auction(targets_row)

        for bidder_number, current_bidder in enumerate(bidder_agents):
            if current_bidder.can_bid:
                noise = np.random.normal(1)  # TODO: find a smarter way!
                bid_value = current_bidder.bid(features_row) + noise
                rtb.receive_new_bid(bid_value)

        pay_price, click = rtb.report_win_notice()

        for bidder_number, current_bidder in enumerate(bidder_agents):
            if current_bidder.can_bid:
                current_bidder.read_win_notice(cost=pay_price, click=click)

                if counter % 100 == 0:
                    print(f"Iteration n {counter}, bidder n {bidder_number}")
        counter += 1

    now = datetime.datetime.now()
    name = 'new_train_' + str(now)
    return rtb.retrieve_new_set_after_auction(set_name=name)
