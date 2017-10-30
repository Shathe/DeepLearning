import argparse
from tensorflow.contrib.keras import optimizers, layers, models, callbacks, utils, preprocessing, metrics

import tensorflow.contrib.keras
from models.ShatheNet_multimodal import *
import numpy as np
import glob
import os
import random
from shutil import copyfile
import cv2
parser = argparse.ArgumentParser()
parser.add_argument("--dataFolder", help="folder where the images are going to be saved")

args = parser.parse_args()
train_data_dir = str(args.dataFolder) + 'train'
validation_data_dir = str(args.dataFolder) + 'test'

# python train_multimodal.py --dataFolder Dataset

epochs = 250
learning_rate = 0.001
n_batches = 32
n_classes = 0
for _, dirnames, filenames in os.walk(train_data_dir):
  # ^ this idiom means "we won't be using this value"
    n_classes += len(dirnames)
    break
x_train, description_train, y_train, x_test, description_test, y_test, dic = getData(dataFolder=str(args.dataFolder), 
															n_training_samples_train = 30000,n_training_samples_test= 5000, n_classes=n_classes)


#preprocessing_function
def preproces(x):
	return x/255.0 - 0.5


x_test = preproces(x_test)
x_train = preproces(x_train)

'''
for a in y_train:
	print(a)
'''
model = ShatheNet_v2_0_multimodal(n_classes=n_classes, weights=None, shape_images=(192, 192, 3), shape_text=description_train.shape[1:])
#model = ShatheNet_text(n_classes=n_classes, weights=None, shape_images=(192, 192, 3), shape_text=description_train.shape[1:])
model.summary()

adam = optimizers.Adam(lr=learning_rate) # decay=0.0001? decay 1/(1+decay*epochs*batches_per_epoch)*lr
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[metrics.categorical_accuracy])
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', verbose=1)

model.fit([x_train, description_train], y_train, batch_size=n_batches, epochs=epochs, validation_data=([x_test, description_test], y_test), callbacks=[reduce_lr])
#model.fit(x_train, y_train, batch_size=n_batches, epochs=epochs, validation_data=(x_test, y_test))
#model.fit(description_train, y_train, batch_size=n_batches, epochs=epochs, validation_data=(description_test, y_test))

score = model.evaluate(x_test, y_test, batch_size=n_batches)
#model.save('my_model_final.h5')
print('Test loss:', score[0])
print('Test accuracy:', score[1])



'''
0.55 solo con imagenes, 0.54 solo con texto. Juntando las redes ->0.75

'''

'''

for doing this with generators... ->https://github.com/fchollet/keras/issues/3386


data_gen_args = dict(preprocessing_function=preproces,
				    rotation_range=35,
				    width_shift_range=0.15,
				    height_shift_range=0.15,
				    shear_range=1.5,
				    zoom_range=0.35,
				    channel_shift_range=0.1,
				    horizontal_flip=True,
				    vertical_flip=True)
# this is the augmentation configuration we will use for training
train_datagen = preprocessing.image.ImageDataGenerator(**data_gen_args)
# Other options:rotation_range, height_shift_range, featurewise_center, vertical_flip, featurewise_std_normalization...
# Also you can give a function as an argument to apply to every iamge

#datagen.fit(x_train)

test_datagen = preprocessing.image.ImageDataGenerator(preprocessing_function=preproces)


train_generator = train_datagen.flow(x_train, y_train, batch_size=n_batches)

validation_generator = test_datagen.flow(x_test, y_test, batch_size=n_batches) 

# fits the model on batches with real-time data augmentation:
model.fit_generator(train_generator,
                    steps_per_epoch=len(x_train) // n_batches, epochs=epochs, validation_data=validation_generator, validation_steps=len(x_test) // n_batches)
'''