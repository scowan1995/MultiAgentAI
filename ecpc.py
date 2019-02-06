from data_loader import DataLoader


class ECPC:
    # Class for calculating the expected cost per click

    def __init__(self):
        dl = DataLoader(load_small=False)
        self.train, self.test = dl.get_train_val(validation_size=0)

    def calculate(self):
        """
        Calculates the expected cost per click
        sum(cost) / num_clicks
        """
        numerator = self.train.loc[self.train["click"] == 1, "payprice"].sum()
        denominator = self.train.loc[self.train["click"] == 1, "click"].sum()
        print(numerator/denominator)

if __name__ == "__main__":
    ecpc = ECPC()
    ecpc.calculate()
