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

CATEGORIES = ["NORMAL","PNEUMONIA"]
model = keras.models.load_model("./best_model.h5")


path =  "./chest_xray/test/NORMAL/"
img_path = glob(path + "*.jpeg")
rand_img = random.sample(img_path,1)[0]

def next():
    category = random.choice(CATEGORIES)
    path =  "./chest_xray/test/" + category + "/"
    img_path = glob(path + "*.jpeg")
    rand_img = random.sample(img_path,1)[0]
    root.img_temp = ImageTk.PhotoImage((Image.open(rand_img)).resize((300,300)))
    image_Lab.configure(image=root.img_temp)
    caption_Lab.configure(text=category)




def model_predict(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image_sized = cv2.resize(img,(64,64))
    X = np.array(image_sized).reshape(-1,64,64,1)/255
    y = int(round(model.predict(X)[0,0],0))
    prediction = CATEGORIES[y]







root = Tk()
root.geometry("300x600")  
root.title("Chest X-ray prediction")
root.iconbitmap("./images/icon.ico")




root.img_init = ImageTk.PhotoImage((Image.open(rand_img)).resize((300,300)))

image_Lab = Label(root,image = root.img_init)
image_Lab.pack()

caption_Lab = Label(root, text = "NORMAL")
caption_Lab.pack()


Next_Btn = Button(root,text="Next Image", command = next)
Next_Btn.pack()
Next_Btn.place(x=10,y=550)

root.mainloop() 
