
import sys
sys.path.append("../../")


from Configs.configs import statics, configs



def scale(data):
    """
    scale numerical data between 0 and 1
    :param data: pandas dataframe with numericalized values
    :return: returns scaled pandas dataframe with numericalized values
    """

    # Importing MinMaxScaler and initializing it
    from sklearn.preprocessing import MinMaxScaler
    min_max = MinMaxScaler()
    # Scaling down both train and test data set
    scaled_data = min_max.fit_transform(data[['weekday', 'hour', 'bidid', 'userid', 'useragent', 'IP', 'region', 'city', 'adexchange', 'domain', 'url', 'urlid', 'slotid', 'slotwidth', 'slotheight', 'slotvisibility', 'slotformat', 'slotprice', 'creative', 'bidprice', 'payprice', 'keypage', 'advertiser', 'usertag']])

    return scaled_data




# main ______________________________________________________________________________________

if __name__ == "__main__":

    print("preprocessing")


