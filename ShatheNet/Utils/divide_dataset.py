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


for data_folder in glob.glob("256_classes_dataset/*"):
	class_name = data_folder[24:len(data_folder)].replace('-101','')
	# Create class folders
	if not os.path.exists(TRAIN_PATH + class_name):
		os.makedirs(TRAIN_PATH  + class_name)
	if not os.path.exists(TEST_PATH  + class_name):
		os.makedirs(TEST_PATH  + class_name)

	i = 1
	for image in glob.glob(data_folder + "/*"):
		folder_to_copy = TRAIN_PATH
		if random.random() > TRAIN_PRCNT:
			folder_to_copy = TEST_PATH
		dst = folder_to_copy + class_name + "/" + str(i) + ".jpg"
		print(image + " to -> " + dst)
		copyfile(image, dst)
		i = i + 1
	print(class_name)
	print(folder_to_copy)
