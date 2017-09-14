import argparse

from tensorflow.contrib.keras import optimizers, layers, models, callbacks, utils, preprocessing
from models.Inception import InceptionModel
from models.ShatheNet import *
from models.ShatheNet_v2 import *
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument("--dataFolder", help="folder where the images are going to be saved")

args = parser.parse_args()
train_data_dir = args.dataFolder + 'train'
validation_data_dir = args.dataFolder + 'test'

epochs = 600
batch_size = 64
learning_rate = 0.00015




#preprocessing_function
def preproces(x):
	# global mean
	# global mean
	x -= np.array([113.75182343,  122.83719635, 125.19327545])
	#x = x/255.0 - 0.5
	return x



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


# this is the augmentation configuration we will use for testing:
test_datagen = preprocessing.image.ImageDataGenerator(preprocessing_function=preproces)

# Generator of images from the data folder
train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(192, 192),
                                                    batch_size=batch_size, class_mode='categorical', shuffle=True)

validation_generator = test_datagen.flow_from_directory(validation_data_dir, target_size=(192, 192),
                                                        batch_size=(batch_size), class_mode='categorical', shuffle=True)

# train the model on the new data for a few epochs


n_classes = train_generator.num_class
nb_train_samples = train_generator.samples 
nb_validation_samples = validation_generator.samples 

#model = InceptionModel(input_tensor=input_tensor, n_classes=n_classes, weights=None, include_top=False)
model = ShatheNet_v2(n_classes=n_classes)

model.summary() 
utils.plot_model(model, to_file='v2.png')


# compile the model (should be done *after* setting layers to non-trainable)
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

adam = optimizers.Adam(learning_rate) # decay=0.0001? decay 1/(1+decay*epochs*batches_per_epoch)*lr
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])


model.fit_generator(train_generator, steps_per_epoch=nb_train_samples // batch_size, epochs=epochs,
                    validation_data=validation_generator, validation_steps=nb_validation_samples // batch_size)

score = model.evaluate_generator(validation_generator, nb_validation_samples)
#model.save('my_model_final.h5')
print('Test loss:', score[0])
print('Test accuracy:', score[1])

