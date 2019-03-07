# test results
import pandas as pd


def stick_together_with_bidid():
    pays = pd.read_csv('pay_predictions_15.csv').astype('int32')
    data = pd.read_csv("Data/validation.csv")
    bids = data.loc[:, "bidid"]
    val_file = pd.concat([bids, pays], axis=1)
    val_file.to_csv('Group_27.csv', header=False, index=False)
    print("writen", val_file.shape)

if __name__ == "__main__":
    stick_together_with_bidid()