import glob
import os
import random
from shutil import copyfile


TRAIN_PRCNT= 0.77
TEST_PATH = "Dataset/test/"
TRAIN_PATH = "Dataset/train/"
# Create train and test folders
if not os.path.exists(TEST_PATH):
    os.makedirs(TEST_PATH)

if not os.path.exists(TRAIN_PATH):
    os.makedirs(TRAIN_PATH)


for filename in glob.glob("Dataset/*/*/*.png"):
	try:
		os.remove(filename[:-4])
	except OSError:
	    pass

