class BasicBiddingAgent:
    def __init__(self, training_set, initial_budget):
        self._current_budget = initial_budget
        self._bid_value = 0
        self._bidding_model = None

        self.can_bid = True
        self.clicks_obtained = 0
        self._total_paid = 0
        self._bids_won = 0

        self._train(training_set)

    def get_current_bid_value(self):
        return self._bid_value

    def get_current_budget(self):
        return self._current_budget

    def get_current_click_through_rate(self):
        ctr = 0
        if self._bids_won > 0:
            ctr = self.clicks_obtained / self._bids_won
        return ctr

    def get_current_cost_per_click(self):
        cpc = 0
        if self.clicks_obtained > 0:
            cpc = self._total_paid / self.clicks_obtained
        return cpc

    def get_bids_won(self):
        return self._bids_won

    def get_total_paied(self):
        return self._total_paid

    def bid(self, ad_user_auction_info=None):
        """Receive a bid request and return the bid"""
        self._process_bid_request(ad_user_auction_info)
        # if bidder doesn't have money to place the bid he would like,
        # it still try to bid all the remained budget hoping in good luck
        return min(self._bid_value, self._current_budget)

    def read_win_notice(self, cost, click=False):
        """Receive the auction outcome and update class instance attributes

        :param
            cost: amount payed for the ad. Since we use a second price auction this value is not the
                same as our bid value (It is the second highest bid value)
                This value is updated by the RTB Ad exchange.
            click: flag that is True when the user clicked on the ad => agent has to pay for the
                impression
        :return True if it wins
        """
        win_flag = False
        if self._bid_value > cost:
            # Bidder wins the auction
            win_flag = True
            self._bids_won += 1
            if click:
                self.clicks_obtained += 1
            # Bidder has to pay impression
            self._total_paid += cost
            self._current_budget -= cost
            if self._current_budget <= 0:
                # Bidder finishes budget and can't bid anymore
                self.can_bid = False

        return win_flag

    def _train(self, training_set):
        """Learn parameters of the bidding model"""
        raise NotImplementedError

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
        raise NotImplementedError
