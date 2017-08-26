import glob
import os
import random
from shutil import copyfile
import cv2
import numpy as np
TRAIN_PRCNT= 0.77
TEST_PATH = "test/"
TRAIN_PATH = "train/"

n_training_samples = 23493
dimension = (224, 224)
channels = 3
join = (n_training_samples,) + dimension + (channels,) 
images = np.zeros(join, dtype=np.uint8)
i = 0
for filename in glob.glob(TRAIN_PATH +"*/*"):
	print(filename)
	img = cv2.imread(filename)
	img = cv2.resize(img, dimension) 
	images[i,:,:,:]=img
	i = i + 1

mean = np.mean(images, axis = 0, dtype=np.float32)
mean = np.mean(mean, axis = 0, dtype=np.float32)
mean = np.mean(mean, axis = 0, dtype=np.float32)
print(mean)
