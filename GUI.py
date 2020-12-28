import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow import keras
from glob import glob
import random
import cv2
import matplotlib.pyplot as plt
from tkinter import *
from PIL import ImageTk, Image


model = keras.models.load_model("./best_model.h5")


path =  "./chest_xray/test/NORMAL/"
img_path = glob(path + "*.jpeg")

rand_img = random.sample(img_path,1)
img = cv2.imread(rand_img[0], cv2.IMREAD_GRAYSCALE)
image_sized = cv2.resize(img,(64,64))
X = np.array(image_sized).reshape(-1,64,64,1)/255

root = Tk()
root.geometry("300x600")  
canvas = Canvas(root, width = 300, height = 300)  
canvas.pack()  
img_temp = ImageTk.PhotoImage((Image.open(rand_img[0])).resize((300,300)))


canvas.create_image(150, 150, image=img_temp) 
root.mainloop() 



print(round(model.predict(X)[0,0],0))
