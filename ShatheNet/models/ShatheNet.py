from keras import backend as K
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

def ShatheNet_v1_0(n_classes=256, weights=None):
    model = Sequential()
    input_shape = (192, 192, 3)
    model.add(Conv2D(16, (7, 7), padding='same', activation='relu', input_shape=input_shape,
                     kernel_initializer='truncated_normal', strides=(6, 6)))
    model.add(Conv2D(16, (5, 5), padding='same', activation='relu', input_shape=input_shape,
                 kernel_initializer='truncated_normal', strides=(4, 4)))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape,
             kernel_initializer='truncated_normal', strides=(2, 2)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    #model.add(Dropout(dropout_rate))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))
    if weights:
        model.load_weights(weights)
    return model


def ShatheNet_v1_1(n_classes=256, weights=None):
    model = Sequential()
    input_shape = (192, 192, 3)
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


def ShatheNet_v1_2(n_classes=256, weights=None):
    model = Sequential()
    input_shape = (192, 192, 3)
    model.add(Conv2D(32, (5, 5), padding='valid', activation='relu', input_shape=input_shape,
                     kernel_initializer='truncated_normal'))
    model.add(Conv2D(32, (5, 5), padding='valid', kernel_initializer='truncated_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='valid', activation='relu', kernel_initializer='truncated_normal'))
    model.add(Conv2D(64, (3, 3), padding='valid', activation='relu', kernel_initializer='truncated_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='valid', activation='relu', kernel_initializer='truncated_normal'))
    model.add(Conv2D(64, (3, 3), padding='valid', activation='relu', kernel_initializer='truncated_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='valid', activation='relu', kernel_initializer='truncated_normal'))
    model.add(Conv2D(128, (3, 3), padding='valid', activation='relu', kernel_initializer='truncated_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), padding='valid', activation='relu', kernel_initializer='truncated_normal'))
    model.add(Conv2D(128, (3, 3), padding='valid', activation='relu', kernel_initializer='truncated_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))


    #model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))
    if weights:
        model.load_weights(weights)
    return model


def ShatheNet_v1_3(n_classes=256, weights=None):
    model = Sequential()
    input_shape = (192, 192, 3)
    model.add(Conv2D(32, (5, 5), padding='same', activation='relu', input_shape=input_shape,
                     kernel_initializer='truncated_normal'))
    model.add(Conv2D(32, (5, 5), padding='same', kernel_initializer='truncated_normal'))
    model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer='truncated_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (5, 5), padding='same', activation='relu', kernel_initializer='truncated_normal'))
    model.add(Conv2D(64, (5, 5), padding='same', activation='relu', kernel_initializer='truncated_normal'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='truncated_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='truncated_normal'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='truncated_normal'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='truncated_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='truncated_normal'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='truncated_normal'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='truncated_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), padding='valid', activation='relu', kernel_initializer='truncated_normal'))
    model.add(Conv2D(128, (3, 3), padding='valid', activation='relu', kernel_initializer='truncated_normal'))
    model.add(Conv2D(128, (3, 3), padding='valid', activation='relu', kernel_initializer='truncated_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))
    if weights:
        model.load_weights(weights)
    return model

    