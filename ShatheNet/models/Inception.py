import argparse

from keras import optimizers
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

def InceptionModel(input_tensor=None, n_classes=256, weights=None, include_top=False):
	if input_tensor != None:
			base_model = InceptionV3(input_tensor=input_tensor, weights=weights, include_top=include_top)

	else:
			base_model = InceptionV3(weights=weights, include_top=include_top)
	# print(base_model.load_weights("my_model_final.h5", by_name=True))

	# add a global spatial average pooling layer
	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	# let's add a fully-connected layer
	x = Dense(1024, activation='relu')(x)

	# and a logistic layer -- let's say we have 4 classes
	predictions = Dense(n_classes, activation='softmax')(x)

	# this is the model we will train
	return Model(inputs=base_model.input, outputs=predictions)



