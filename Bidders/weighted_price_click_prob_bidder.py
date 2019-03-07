import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


class LinearRegressionBidder():

    def __init__(self):
        X, Y = self.get_data()
        self.trainx, self.testx, self.trainy, self.testy = train_test_split(X, Y, test_size=0.1)
        self.model = LinearRegression(normalize=True, n_jobs=-1)

    def train(self):
        return self.model.fit(self.trainx, self.trainy)

    def test(self):
        print("model score:", self.model.score(self.testx, self.testy))

    def get_data(self):
        click_probs = np.genfromtxt("click_predictions.csv", delimiter=',')[:, 1]
        pay_preds = np.genfromtxt("pay_predictions_15.csv")
        data = pd.DataFrame(np.concatenate((click_probs, pay_preds), axis=1))
        print("data shape", data.shape)
        real_pay = pd.read_csv("Data/train.csv")
        print(data)
        return data, real_pay

if __name__ == "__main__":
    lb = LinearRegressionBidder()
    print("training")
    lb.train()
    print("testing")
    lb.test()

        