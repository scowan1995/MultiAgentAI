
## ________________________________________________
##
## Neural Network for Encoding

import shutil
import os, sys
import pickle
import numpy.random as rng
import numpy as np
from sklearn.utils import shuffle
import random

# do not comment back in when using a server, since it causes issues otherwise!
# import matplotlib.pylab as plt

import sys
sys.path.append("../")
sys.path.append("../../")

from Configs.configs import statics, configs


# importing different models:
from keras.applications import VGG19, ResNet50, DenseNet121, NASNetLarge, MobileNet
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras import regularizers, initializers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.models import Model, Input
import numpy as np
import matplotlib.pylab as plt
from random import shuffle
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import model_from_json

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import warnings
warnings.filterwarnings('ignore', '.*do not.*',)




def create_model(input_shape, nClasses):

    ## sequential model
    # model = Sequential()
    #
    # model.add(Dense(2**5, input_dim=input_shape))
    # model.add(Dropout(0.5))
    #
    # ## 4096 like convnet output
    # model.add(Dense(2**5, activation='relu'))
    # model.add(Dropout(0.1))
    # model.add(Dense(nClasses, activation='softmax'))

    inputs = Input(shape=(input_shape,))
    predictions = Dense(nClasses, activation="softmax", kernel_regularizer=regularizers.l2(5e-4),kernel_initializer=initializers.he_normal(seed=13))(inputs)
    model = Model(inputs=inputs, outputs=predictions, name='semantics_encoder')

    return model



def plot_curves(history):

    # Loss Curves
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['loss'], 'r', linewidth=3.0)
    plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Loss Curves', fontsize=16)
    plt.savefig(statics['result'] + '/cnn_loss_curve.png')
    #plt.show()

    # Accuracy Curves
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['acc'], 'r', linewidth=3.0)
    plt.plot(history.history['val_acc'], 'b', linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy Curves', fontsize=16)
    plt.savefig(statics['result']  + '/cnn_accuracy_curve.png')
    #plt.show()






def prepare_data_for_classification(feature_vectors, class_labels):

    ## prepare

    ## image_data
    feature_vectors_list = np.asarray(list(feature_vectors.values()))

    ## labels
    class_labels_list = np.asarray(list((map(int, class_labels.keys()))))

    ## shuffle
    # indices = np.arange(semantic_vectors_list.shape[0])
    #
    # np.random.shuffle(indices)
    # class_labels_list = class_labels_list[indices]
    # semantic_vectors_list = semantic_vectors_list[indices]

    return feature_vectors_list, class_labels_list







if __name__ == "__main__":


    ## prepare data for classification task including shuffle
    ## note that training classes = validation classes = testing classes due to classification task!


   # data, labels_list = load_data....

    nClasses = data.shape[0]
    input_shape = data.shape[1]


    print("vector input shape: ", input_shape, " number of classes for classification: ", nClasses)


    ## check if still correct data
    #check_images(data, labels_list)

    # Clear model, and create it
    # model = None
    model = create_model(input_shape, nClasses)
    print(model.summary())


    ## save model to disk_____________________________________

    # serialize model to JSON
    #model_json = model.to_json()
    #if not os.path.exists(static_paths['data_path'] + '/trained_classifiers'):
    #    os.mkdir(static_paths['data_path'] + '/trained_classifiers')

    #with open(static_paths['data_path'] + '/trained_classifiers' + '/semantic_encoder_architecture.json', "w") as json_file:
    #    json_file.write(model_json)

    labels_one_hot = to_categorical(labels_list, num_classes= max(labels_list)+1 )


    ## model compile and train
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Set callback functions to early stop training and save the best model so far
    callbacks = [EarlyStopping(monitor='loss', patience=200),
                 ModelCheckpoint(filepath= static_paths['data_path'] + '/trained_classifiers' + '/semantic_encoder_weights.h5', monitor='loss', save_best_only=True)]

    # Fit the model on the batches
    batch_size = 12
    epochs = 120

    history = model.fit(x=data, y=labels_one_hot, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=callbacks, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None)

    ## model evaluate
    #plot_curves(history)



    ## Load Model for Testing_______________________________________________________________

    # Model reconstruction from JSON file
    #with open(static_paths['data_path'] + '/trained_classifiers' + '/semantic_encoder_architecture.json', 'r') as f:
    #    model_test = model_from_json(f.read())

    # Load weights into the new model
    #model_test.load_weights(static_paths['data_path'] + '/trained_classifiers' + '/semantic_encoder_weights.h5')

    #model_test.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    #test_score = model_test.evaluate(x = data, y = to_categorical(labels_list, num_classes= max(labels_list)+1))
    #print('\n\nTest accuracy:', test_score[1])


