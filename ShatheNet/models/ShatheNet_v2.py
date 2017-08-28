from tensorflow.contrib.keras  import backend as K

from tensorflow.contrib.keras import optimizers, layers, models, callbacks, utils, preprocessing

import numpy as np

def ShatheNet_v2_0(n_classes=256, weights=None):
    # paddign same, filtros mas pequemos.. 
    input_shape = (192, 192, 3)

    inputs = layers.Input(shape=input_shape)

    # a layer instance is callable on a tensor, and returns a tensor
    x = layers.Conv2D(16, (7, 7), padding='same', activation='relu', input_shape=input_shape,
                     kernel_initializer='truncated_normal', strides=(6, 6))(inputs)
    x = conv2d_bn(inputs, 16, 5, 5, padding='same', strides=(2, 2))
    x = conv2d_bn(x, 32, 1, 1, padding='same', strides=(2, 2))
    x = conv2d_bn(x, 32, 3, 3, padding='same', strides=(2, 2))
    x = layers.MaxPooling2D((3, 3))(x)
    x = dense_block(x, 8, 16)
    x = layers.GlobalAveragePooling2D()(x)
    # x = layers.Flatten()(x)
    predictions = layers.Dense(n_classes, activation='softmax')(x)

    # This creates a model that includes
    # the Input layer and three Dense layers
    model = models.Model(inputs=inputs, outputs=predictions)
    if weights:
        model.load_weights(weights)
    return model

def ShatheNet_v2_1(n_classes=256, weights=None):
    # paddign same, filtros mas pequemos.. 
    input_shape = (192, 192, 3)

    inputs = layers.Input(shape=input_shape)

    # a layer instance is callable on a tensor, and returns a tensor
    x = layers.Conv2D(16, (7, 7), padding='same', activation='relu', input_shape=input_shape,
                     kernel_initializer='truncated_normal', strides=(6, 6))(inputs)
    x = conv2d_bn(inputs, 32, 5, 5, padding='same', strides=(2, 2))
    x = conv2d_bn(x, 64, 1, 1, padding='same', strides=(1, 1))
    x = conv2d_bn(x, 64, 3, 3, padding='same', strides=(1, 1))
    x = layers.MaxPooling2D((2, 2))(x)
    x = dense_block(x, 4, 16)
    x = transition_block(x, 64)
    x = dense_block(x, 8, 16)
    x = transition_block(x, 64)
    x = dense_block(x, 12, 24)
    x = transition_block(x, 128)
    x = dense_block(x, 16, 32)
    x = transition_block(x, 128)
    x = dense_block(x, 12, 24)
    x = layers.GlobalAveragePooling2D()(x)
    # x = layers.Flatten()(x)
    predictions = layers.Dense(n_classes, activation='softmax')(x)

    # This creates a model that includes
    # the Input layer and three Dense layers
    model = models.Model(inputs=inputs, outputs=predictions)
    if weights:
        model.load_weights(weights)
    return model

def ShatheNet_v2_2(n_classes=256, weights=None):
    # paddign same, filtros mas pequemos.. 
    input_shape = (192, 192, 3)

    inputs = layers.Input(shape=input_shape)

    # a layer instance is callable on a tensor, and returns a tensor
    x = layers.Conv2D(16, (7, 7), padding='same', activation='relu', input_shape=input_shape,
                     kernel_initializer='truncated_normal', strides=(6, 6))(inputs)
    x = conv2d_bn(inputs, 32, 5, 5, padding='same', strides=(2, 2))
    x = conv2d_bn(x, 64, 1, 1, padding='same', strides=(1, 1))
    x = conv2d_bn(x, 64, 3, 3, padding='same', strides=(1, 1))
    x = layers.MaxPooling2D((3, 3))(x)
    x = dense_block(x, 6, 72)
    x = transition_block(x, 64)
    x = dense_block(x, 12, 72)
    x = transition_block(x, 128)
    x = dense_block(x, 8, 72)
    x = layers.GlobalAveragePooling2D()(x)
    # x = layers.Flatten()(x)
    predictions = layers.Dense(n_classes, activation='softmax')(x)

    # This creates a model that includes
    # the Input layer and three Dense layers
    model = models.Model(inputs=inputs, outputs=predictions)
    if weights:
        model.load_weights(weights)
    return model
def ShatheNet_v2_3(n_classes=256, weights=None):
    # paddign same, filtros mas pequemos.. 
    input_shape = (192, 192, 3)

    inputs = layers.Input(shape=input_shape)

    # a layer instance is callable on a tensor, and returns a tensor
    x = layers.Conv2D(16, (7, 7), padding='same', activation='relu', input_shape=input_shape,
                     kernel_initializer='truncated_normal', strides=(6, 6))(inputs)
    x = conv2d_bn(inputs, 32, 5, 5, padding='same', strides=(2, 2))
    x = conv2d_bn(x, 64, 1, 1, padding='same', strides=(1, 1))
    x = conv2d_bn(x, 64, 3, 3, padding='same', strides=(1, 1))
    x = layers.MaxPooling2D((2, 2))(x)
    x = dense_block(x, 8, 24)
    x = transition_block(x, 64)
    x = dense_block(x, 12, 16)
    x = transition_block(x, 128)
    x = dense_block(x, 18, 16)
    x = transition_block(x, 128)
    x = dense_block(x, 12, 16)
    x = layers.GlobalAveragePooling2D()(x)
    # x = layers.Flatten()(x)
    predictions = layers.Dense(n_classes, activation='softmax')(x)

    # This creates a model that includes
    # the Input layer and three Dense layers
    model = models.Model(inputs=inputs, outputs=predictions)
    if weights:
        model.load_weights(weights)
    return model







def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1)):
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3

    x = layers.Conv2D(filters, (num_row, num_col), strides=strides, padding=padding)(x) # use_bias=False,
    x = layers.BatchNormalization(axis=bn_axis)(x) # scale=False,
    x = layers.Activation('relu')(x)
    return x

def node(x, nb_filter):

    tower_1 = conv2d_bn(x, nb_filter, 1, 1, padding='same', strides=(1, 1))
    tower_1 = conv2d_bn(tower_1, nb_filter, 3, 3, padding='same', strides=(1, 1))

    tower_2 = conv2d_bn(x, nb_filter, 1, 1, padding='same', strides=(1, 1))
    tower_2 = conv2d_bn(tower_2, nb_filter, 5, 5, padding='same', strides=(1, 1))

    output = layers.concatenate([tower_1, tower_2], axis=3)
    return output
    #ahora toca un denseblock con los nodos


def dense_block(x, nb_layers, nb_filter):
    #Hacer algo como contaenation imapres o pares para ver si se reduce el numero mucho?
  
    concatetation_of_inputs = x
    for i in range(nb_layers):
        next_node = node(concatetation_of_inputs, nb_filter)
        concatetation_of_inputs = layers.concatenate([concatetation_of_inputs, next_node], axis=3)
        previous_node = next_node
        nb_filter = nb_filter + 8


    return concatetation_of_inputs #hacia la transition layer

def transition_block(x, nb_filter):

    x = conv2d_bn(x, nb_filter, 1, 1, padding='same', strides=(1, 1))
    x = layers.AveragePooling2D((2, 2), strides=(2, 2))(x)
    return x



    '''


Cada bloque tiene distinto numero de laters por ejemplo 6,12,24,16 y el growth rate es lo que crece, empezando por 32 por ejemplo, en cada una de esas capas 

Poner algo asi ya que son muchas capas , y no hacer lo de par o impar
'''


