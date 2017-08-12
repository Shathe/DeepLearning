from keras import backend as K
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

def ShatheNet_v1(n_classes=256, weights=None):
    model = Sequential()
    input_shape = (299, 299, 3)
    model.add(Conv2D(32, (7, 7), padding='same', activation='relu', input_shape=input_shape,
                     kernel_initializer='truncated_normal', strides=(2, 2)))
    model.add(Conv2D(32, (7, 7), padding='same', kernel_initializer='truncated_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))
    model.add(Conv2D(64, (5, 5), padding='same', activation='relu', kernel_initializer='truncated_normal', strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), padding='same', activation='relu', kernel_initializer='truncated_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='truncated_normal', strides=(2, 2)))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='truncated_normal'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    #model.add(Dropout(dropout_rate))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))
    if weights:
        model.load_weights(weights)
    return model

'''
Use keras funcitonal

# This returns a tensor
inputs = Input(shape=(784,))

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
'''
def ShatheNet_v2(n_classes=256, weights=None):
        # This returns a tensor
    inputs = Input(shape=(784,))

    # a layer instance is callable on a tensor, and returns a tensor
    x = Dense(64, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(10, activation='softmax')(x)

    # This creates a model that includes
    # the Input layer and three Dense layers
    model = Model(inputs=inputs, outputs=predictions)
    return model