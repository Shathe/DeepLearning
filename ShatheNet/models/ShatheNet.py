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



def conv_block_v1(x, nb_filter, nb_row, nb_col, border_mode='same', subsample=(1, 1), bias=False):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    x = Convolution2D(nb_filter, nb_row, nb_col, subsample=subsample, border_mode=border_mode, bias=bias)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    return x

def conv_block_v2(x, nb_filter, nb_row, nb_col, border_mode='same', subsample=(1, 1), bias=False):
    model_layers[level]['b_norm'+str(i+1)] = BatchNormalization(mode=0, axis=3,
                                         gamma_regularizer=l2(0.0001),
                                         beta_regularizer=l2(0.0001))(previous_layer)
    model_layers[level]['act'+str(i+1)] = Activation('relu')(model_layers[level]['b_norm'+str(i+1)])
    model_layers[level]['conv'+str(i+1)] = Conv2D(filters,   kernel_size=(3, 3), padding='same',
                                kernel_initializer="he_uniform",
                                data_format='channels_last')(model_layers[level]['act'+str(i+1)])
    model_layers[level]['drop_out'+str(i+1)] = Dropout(0.2)(model_layers[level]['conv'+str(i+1)])
    previous_layer  = model_layers[level]['drop_out'+str(i+1)]
    return model_layers[level]['drop_out'+ str(layers_count)]

    