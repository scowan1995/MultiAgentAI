
class RtbAdExchange:
    def __init__(self):
        self._bids = []
        self._cost = 0
        self._click = False

    def evaluate_known_auction(self, known_auction_outcome):
        """ Interpret results of a finished auction

        If the evaluation of a known auction is required, this function has to be called before
        other agents start joining the auction.

        :param known_auction_outcome: this is a targets row, which can come from the validation sets
        """
        self._bids = []
        self._bids.append(known_auction_outcome['bidprice'])
        self._cost = known_auction_outcome['payprice']
        self._click = bool(known_auction_outcome['click'])

    def receive_new_bid(self, bidder_bid):
        self._bids.append(bidder_bid)

    def generated_a_click(self):
        """Manually decide if it generated a click.

        It can be useful if there are no information from previous auctions (from a validation set)
        """
        self._click = True

    def report_win_notice(self):
        number_of_bids = len(self._bids)
        assert number_of_bids > 0

        if number_of_bids > 1:
            self._bids.sort()
            # update the cost with second highest bid (we are in a second price auction)
            self._cost = self._bids[-2]

        return self._cost, self._click
