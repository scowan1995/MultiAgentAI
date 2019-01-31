
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

        'mock': True,

        # --- Logistic Regression ---

        'val_ratio': 0.2,

    }

    return statics, configs


def configure_gpu(configs):

    os.environ['KERAS_BACKEND'] = 'tensorflow'  # 'theano' or 'tensorflow
    os.environ["THEANO_FLAGS"] = 'mode=FAST_RUN,device=' + 'gpu' + str(configs['gpu_no']) + ',floatX=float32,force_device=True'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(configs['gpu_no'])



statics, configs = make_statics_configs()
