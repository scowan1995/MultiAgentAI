class BasicBiddingAgent:
    def __init__(self, training_set, initial_budget):
        self._current_budget = initial_budget
        self._bid_value = 0
        self._bidding_model = None

        self._train(training_set)

    def bid(self, ad_user_auction_info=None):
        """Receive a bid request and return the bid"""
        self._process_bid_request(ad_user_auction_info)
        return self._bid_value

    def read_win_notice(self, cost, win=False):
        """Receive the auction outcome and update class instance attributes

        Args:
            cost: amount payed for the ad. Since we use a second price auction this value is not the
                same as our bid value (It is the second highest bid value)
            win: flag that is True when the agent win the auction
        """
        if win:
            self._current_budget -= cost

    def _train(self, training_set):
        """Learn parameters of the bidding model"""
        pass

    def _process_bid_request(self, ad_user_auction_info=None):
        """Process the bid request that is sent to the Bidding Agent

        When a bid request arrives, process the data and choose the amount you want to bid using the
        trained bidding model.

        Args:
            ad_user_auction_info: data type to be defined. It contains information about the ad, the
            user and the auction (ex. hour, region, device, ad size,...)
        """
        self._bid_value = self._bidding_function()

    def _bidding_function(self, utility=None, cost=None):
        """Deploy the learned bidding model"""
        pass
