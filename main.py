import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow import keras
import numpy as np

from matplotlib.image import imread
from glob import glob

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

TEST_LOC = "./chest_xray/test"
TRAIN_LOC = "./chest_xray/train"

"""
N --> Normal
P --> Pneumonia
"""

TRAIN_LOC_N = TRAIN_LOC + "/NORMAL/"
TRAIN_LOC_P = TRAIN_LOC + "/PNEUMONIA/"

IMG_P = glob(TRAIN_LOC_P + "*.jpeg")
IMG_N = glob(TRAIN_LOC_N + "*.jpeg")


import random
import cv2
import matplotlib.pyplot as plt
BATCH_SIZE = 32
IMG_SIZE = 64*2

train_set = []

for im in IMG_P:
    image = cv2.imread(im, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (IMG_SIZE,IMG_SIZE))
    
    train_set.append([image,1])
    
for im in IMG_N:
    image = cv2.imread(im, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (IMG_SIZE,IMG_SIZE))
    
    train_set.append([image,0])

random.shuffle(train_set)

