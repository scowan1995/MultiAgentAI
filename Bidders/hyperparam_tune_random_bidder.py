from random_bidding_agent import RandomBiddingAgent

r = RandomBiddingAgent(0, 6250)
r._train(None)
r.tune(60)
