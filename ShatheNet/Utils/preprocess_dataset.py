import glob
import os
import random
from shutil import copyfile
import cv2
import numpy as np
TRAIN_PRCNT= 0.77
TEST_PATH = "Dataset/test/"
TRAIN_PATH = "Dataset/train/"

n_training_samples = 50000
dimension = (32, 32, 3)
join = (samples,) + dimension
images = np.zeros(join ,dtype=np.float64)
i = 0
for filename in glob.glob(TRAIN_PATH +"*/*"):
	print(filename)
	img = cv2.imread(filename)
	images[i,:,:,:]=img
	i = i + 1



mean = np.mean(images, axis = 0, dtype=np.float64)
mean2=np.array(mean, dtype=np.float64)
#cv2.imwrite('mean.png',mean2) with uint8

std = np.std(images, axis = 0, dtype=np.float64)
std2=np.array(std, dtype=np.float64)
#cv2.imwrite('std.png',std2)  with uint8
std2 += 1e-7

np.save("std", std2)
np.save("mean", mean2)
