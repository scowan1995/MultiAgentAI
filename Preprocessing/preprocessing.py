
import sys
sys.path.append("../../")


from Configs.configs import statics, configs
from sklearn.preprocessing import LabelEncoder


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




def numericalize(data):
    """
    numericalizes categorical columns in pandas dataframe
    :param data: pandas dataframe
    :return: numericalized dataframe and mapping between numbers and previous categorical values
    """
    le = LabelEncoder()
    le_mapping = dict()

    for col in data.columns.values:

        # Encoding only categorical variables
        if data[col].dtypes == 'object':
            # Using whole data to form an exhaustive list of levels

            categoricals = data[col].append(data[col])
            le.fit(categoricals.values.astype(str))
            ## safe mapped data
            data[col] = le.transform(data[col].astype(str))
            ## safe mapping
            le_mapping[col] = dict(zip(le.classes_, le.transform(le.classes_)))

    return data, le_mapping




# main ______________________________________________________________________________________

if __name__ == "__main__":

    print("preprocessing")


