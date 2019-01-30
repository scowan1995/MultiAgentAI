import time
import re
import os
import pickle


"""
configs: List of configs the user mainly interacts with, such as hyperparameters, choosing the dataset etc.
"""


def make_static_configs():

    configs = {

        'local_or_server': 'local',  # 'local' OR 'server'
        'gpu_no': 0,  # integer 0-3    # select the GPU card
        'data_folder': 'Data',
        'result_folder': 'Results',



        # --- TRAINING ---

        # --- VALIDATION ---

        # --- TESTING ---

    }


def configure_gpu(configs):

    os.environ['KERAS_BACKEND'] = 'tensorflow'  # 'theano' or 'tensorflow
    os.environ["THEANO_FLAGS"] = 'mode=FAST_RUN,device=' + 'gpu' + str(configs['gpu_no']) + ',floatX=float32,force_device=True'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(configs['gpu_no'])



configs = make_static_configs()
