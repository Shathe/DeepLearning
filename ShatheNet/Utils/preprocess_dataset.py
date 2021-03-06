import glob
import os
import random
from shutil import copyfile
import cv2
import numpy as np
TRAIN_PRCNT= 0.77
TEST_PATH = "test/"
TRAIN_PATH = "train/"

n_training_samples = 50000
dimension = (192, 192)
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



#IT CAN BE DONE ONLY FOR ONE VALUE PER CHANNEL
'''
print("Computing std...")
std = np.std(images, axis = 0, dtype=np.float32)
std2=np.array(std, dtype=np.float16)
#cv2.imwrite('std.png',std2)  with uint8
std2 += 1e-7

print("Computing mean...")
mean = np.mean(images, axis = 0, dtype=np.float32)
mean2=np.array(mean, dtype=np.float16)
#cv2.imwrite('mean.png',mean2) with uint8

print("Saving...")
np.save("std", std2)
np.save("mean", mean2)

print("Reading...")
mean =  np.load("./mean.npy")
std =  np.load("./std.npy")
print(std.shape)
print(std)

'''