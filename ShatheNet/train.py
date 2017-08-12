import argparse

from keras import optimizers
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from models.Inception import InceptionModel
from models.ShatheNet import ShatheNet

ShatheNet
parser = argparse.ArgumentParser()
parser.add_argument("--dataFolder", help="folder where the images are going to be saved")

args = parser.parse_args()
train_data_dir = args.dataFolder + 'train'
validation_data_dir = args.dataFolder + 'test'
nb_train_samples = 23751 
nb_validation_samples = 7371 

epochs = 30
batch_size = 8
learning_rate = 0.001

import os
n_classes = 0

for _, dirnames, _ in os.walk(train_data_dir):
  # ^ this idiom means "we won't be using this value"
    n_classes += len(dirnames)

#model = InceptionModel(n_classes=n_classes, weights=None, include_top=False)
model = ShatheNet(n_classes=n_classes)

model.summary() 


# compile the model (should be done *after* setting layers to non-trainable)
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

adam = optimizers.adam(learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, width_shift_range=0.15, height_shift_range=0.15, zoom_range=0.35, rotation_range=45)
# Other options:rotation_range, height_shift_range, featurewise_center, vertical_flip, featurewise_std_normalization...
# Also you can give a function as an argument to apply to every iamge


# this is the augmentation configuration we will use for testing:
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Generator of images from the data folder
train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(299, 299),
                                                    batch_size=batch_size, class_mode='categorical', shuffle=True)

validation_generator = test_datagen.flow_from_directory(validation_data_dir, target_size=(299, 299),
                                                        batch_size=(batch_size/2), class_mode='categorical', shuffle=True)

# train the model on the new data for a few epochs


model.fit_generator(train_generator, steps_per_epoch=nb_train_samples // batch_size, epochs=epochs,
                    validation_data=validation_generator, validation_steps=nb_validation_samples // batch_size)

score = model.evaluate_generator(validation_generator, nb_validation_samples)
#model.save('my_model_final.h5')
print('Test loss:', score[0])
print('Test accuracy:', score[1])
