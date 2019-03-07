import os

from Preprocessing import single_set


class RtbAdExchange:
    def __init__(self, set_to_update=None):
        self._bids = []
        self._cost = 0
        self._click = False

        self._row_counter = 0
        self._new_set = None
        if set_to_update is not None:
            self._new_set = set_to_update.data.copy()
            self._new_set['bidprice'] = 0
            self._new_set['payprice'] = 0

    def evaluate_known_auction(self, known_auction_outcome):
        """ Interpret results of a finished auction

        If the evaluation of a known auction is required, this function has to be called before
        other agents start joining the auction.

        :param known_auction_outcome: this is a targets row, which can come from the validation sets
        """
        self._bids = []
        self._bids.append(known_auction_outcome['payprice'])
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

            # insert payprice and bidprice in the new set
            if self._new_set is not None:
                self._new_set.loc[
                    self._new_set.index[self._row_counter], 'payprice'] = self._cost
                self._new_set.loc[
                    self._new_set.index[self._row_counter], 'bidprice'] = self._bids[-1]
                self._row_counter += 1

        return self._cost, self._click

    def retrieve_new_set_after_auction(self, set_name):
        path = "Data/" + set_name + "0000"
        return single_set.SingleSet(relative_path=path, data=self._new_set)
