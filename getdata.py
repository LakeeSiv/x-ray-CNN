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
def make_traindata(condition = False):
    if condition == True:
        for category in CATEGORIES:
            path = TRAIN_LOC + category +"/"
            label = CATEGORIES.index(category)
            for img in glob(path + "*.jpeg"):
                image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
                image_sized = cv2.resize(image,(IMG_SIZE,IMG_SIZE))

                train_set.append([image_sized,label])

        random.shuffle(train_set)

        X,y=[],[]

        for features,label in train_set:
            X.append(features)
            y.append(label)

        X = np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)

        np.save("X_train.npy",X)
        np.save("y_train.npy",y)




make_traindata(condition = False)











"""

for im in random.sample(IMG_P,int(SAMPLE_SIZE/2)):
    
    image = cv2.imread(im, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (IMG_SIZE,IMG_SIZE))
    
    train_set.append([image,1])
   

for im in random.sample(IMG_N,int(SAMPLE_SIZE/2)):
  
    image = cv2.imread(im, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (IMG_SIZE,IMG_SIZE))
    
    train_set.append([image,0])
   

random.shuffle(train_set)


X,y=[],[]

for features,label in train_set:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)

print(y)
"""