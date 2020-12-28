import os
import numpy as np
from matplotlib.image import imread
from glob import glob
import random
import cv2
import matplotlib.pyplot as plt


TEST_LOC = "./chest_xray/test/"
TRAIN_LOC = "./chest_xray/train/"

"""
N --> Normal
P --> Pneumonia
"""
CATEGORIES = ["NORMAL","PNEUMONIA"]

IMG_SIZE = 64


train_set = []

#get train data
def make_data(condition = False, LOC=0):
    if condition == True:
        for category in CATEGORIES:
            path = LOC + category +"/"
            label = CATEGORIES.index(category)
            for img in glob(path + "*.jpeg"):
                image = cv2.imread(img, cv2.IMREAD_GRAYSCALE) #<------ Convert to greyscale
                image_sized = cv2.resize(image,(IMG_SIZE,IMG_SIZE))

                train_set.append([image_sized,label])
                

        random.shuffle(train_set)

        X,y=[],[]

        for features,label in train_set:
            X.append(features)
            y.append(label)

        X = np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)


        if LOC == TRAIN_LOC:
            sub = "train"
        elif LOC == TEST_LOC:
            sub = "test"

        np.save(f"X_{sub}.npy",X)
        np.save(f"y_{sub}.npy",y)


make_data(condition = False, LOC = TEST_LOC)

