import os


"""
configs: List of configs the user mainly interacts with, such as hyperparameters, choosing the dataset etc.
"""

def make_statics_configs():

    statics = {

        'local_or_server': 'local',  # 'local' OR 'server'
        'gpu_no': 0,  # integer 0-3    # select the GPU card
        'data': {'mock': '/Data/mock.csv', 'train': '/Data/train.csv', 'val': '/Data/validation.csv', 'test': '/Data/test.csv'},
        'result': '/Results',

    }



    configs = {

        # --- Dataset and Loading ---

        # which files to load
        'sets': {'mock': True, 'train': False, 'val': False, 'test': False},

        #combine files?
        'combine_train_val': False,
        # load train_val pickle?
        'load_pickle': True,    # load full train and val set, but pickled and preprocessed version

        # target and feature variables
        'target' : 'click',     # 'payprice', 'bidprice'   # select the target
        'features': ['weekday', 'hour', 'bidid', 'userid', 'useragent', 'IP', 'region', 'city', 'adexchange', 'domain', 'url', 'urlid', 'slotid', 'slotwidth', 'slotheight', 'slotvisibility', 'slotformat', 'slotprice', 'creative', 'keypage', 'advertiser', 'usertag'], # select the data features


        # --- Data Exploration ---

        'data_exploration': True,


        # --- Logistic Regression ---

        'logistic_regression': False,    # perform logistic regression
        'val_ratio': 0.2,               # ratio train and validation set


        # --- Neural Network ---

    }

    return statics, configs




def configure_gpu(configs):

    os.environ['KERAS_BACKEND'] = 'tensorflow'  # 'theano' or 'tensorflow
    os.environ["THEANO_FLAGS"] = 'mode=FAST_RUN,device=' + 'gpu' + str(configs['gpu_no']) + ',floatX=float32,force_device=True'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(configs['gpu_no'])


statics, configs = make_statics_configs()
