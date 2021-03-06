import sys

from Main.bidders_rtb_interactions import *
from Preprocessing.preprocessing import *
from Preprocessing.data_exploration import *
from Configs.configs import configs
from Model.logistic_regression import Logistic_Regression
from Bidders.constant_bidding_agent import ConstantBiddingAgent
from Bidders.random_bidding_agent import RandomBiddingAgent
from Bidders.budget_aware_logistic_regression_bidding_agent import BudgetAwareLogisticRegressionBiddingAgent
from Rtb.rtb_ad_exchange import RtbAdExchange
# from Model.neural_network import

sys.path.append("../")


def main():
    # DATA_________________________________________________________________________
    sets_information = configs['sets']
    sets = load_all_datasets(sets_information)
    scale_all_sets_features(sets)
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
        random_bidder = RandomBiddingAgent(training_set=sets['train'],
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

    if configs['budget_aware_logistic_regression']:
        rtb = RtbAdExchange(sets['test'])
        log_reg_bidder = BudgetAwareLogisticRegressionBiddingAgent(training_set=sets['train'],
                                                                   additional_set=sets['test'],
                                                                   initial_budget=bidder_budget)
        log_reg_bidder.set_campaign_duration_from_set(sets['test'])

        # For plotting only, then choose one!
        # gamma_train, min_x, max_x = log_reg_bidder.fit_marketprice_gamma_distribution(sets['train'], plot=False)
        # lognorm_train, _, _ = log_reg_bidder.fit_marketprice_log_normal_distribution(sets['train'], plot=False)
        #
        # gamma_val, min_x_val, max_x_val = log_reg_bidder.fit_marketprice_gamma_distribution(sets['val'], plot=False)
        # lognorm_val, _, _ = log_reg_bidder.fit_marketprice_log_normal_distribution(sets['val'], plot=False)
        #
        # market_prices = [np.asarray(sets['train'].data_targets['payprice']),
        #                  np.asarray(sets['val'].data_targets['payprice'])]
        # functions_range = np.linspace(min(min_x, min_x_val), max(max_x, max_x_val), 100)
        # functions = [gamma_train(functions_range), lognorm_train(functions_range),
        #              gamma_val(functions_range), lognorm_val(functions_range)]
        # plot_multiple_functions_and_distributions(functions_range, functions, market_prices, "distributions1")

        # For validation, train or mock
        # single_agent_interact_with_rtb(log_reg_bidder, rtb, sets, print_results=True)

        # For testing
        single_agent_interact_with_rtb_for_testing(log_reg_bidder, rtb, sets, print_results=True)
        rtb.generate_submission_file()

    if configs['multiple_budget_aware']:
        rtb = RtbAdExchange(sets['val'])
        total_bidders = 20
        first_bidder = BudgetAwareLogisticRegressionBiddingAgent(training_set=sets['val'],
                                                                 additional_set=sets['val'],
                                                                 initial_budget=bidder_budget)
        first_bidder.set_campaign_duration_from_set(sets['val'])

        bidders = [first_bidder]
        for i in range(total_bidders):
            log_reg_bidder = BudgetAwareLogisticRegressionBiddingAgent(training_set=sets['val'],
                                                                       additional_set=sets['val'],
                                                                       initial_budget=bidder_budget,
                                                                       train_flag=False)
            log_reg_bidder.set_logistic_regressor(first_bidder.get_trained_logistic_regressor())
            log_reg_bidder.set_marketprice_upperbound(first_bidder.get_marketprice_upperbound())
            log_reg_bidder.set_campaign_duration_from_set(sets['val'])
            log_reg_bidder.set_click_predictions(first_bidder.get_click_predictions())
            # log_reg_bidder.fit_and_show_marketprice_gamma_distribution(sets['train'], plot=False)
            bidders.append(log_reg_bidder)
        multiagent_bidders_interact_with_rtb_to_generate_new_set(bidders, rtb, sets)
        print("new dataset created")

    if configs['try_to_fit_marketprice_distributions']:
        # Explore marketprice distributions
        new_validation = SingleSet(relative_path='/Data/new_validation_1.csv',
                                   use_numerical_labels=True)
        outlier_threshold = 600
        log_reg_bidder = BudgetAwareLogisticRegressionBiddingAgent(training_set=sets['train'],
                                                                   additional_set=sets['test'],
                                                                   initial_budget=bidder_budget,
                                                                   train_flag=True)

        gamma_old, min_x, max_x = log_reg_bidder.fit_marketprice_gamma_distribution(sets['val'], plot=False)
        lognorm_old, _, _ = log_reg_bidder.fit_marketprice_log_normal_distribution(sets['val'], plot=False)

        new_payprices = np.asarray(new_validation.data['payprice'])
        market_prices = [np.asarray(sets['val'].data_targets['payprice']),
                         new_payprices[new_payprices < outlier_threshold]]
        functions_range = np.linspace(min_x, max_x, 100)
        functions = [gamma_old(functions_range), lognorm_old(functions_range)]
        plot_multiple_functions_and_distributions(functions_range, functions, market_prices, "fitted_distributions_3")


if __name__ == "__main__":
    main()
    plt.show()
