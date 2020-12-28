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




def next():
    global category 
    category = random.choice(CATEGORIES)
    path =  "./chest_xray/test/" + category + "/"
    img_path = glob(path + "*.jpeg")
    global rand_img_path 
    rand_img_path = random.sample(img_path,1)[0]
    root.img_temp = ImageTk.PhotoImage((Image.open(rand_img_path)).resize((300,300)))
    image_Lab.configure(image=root.img_temp)

    Next_Btn.configure(text = "Next Image")
    Predict_Btn.configure(text = "Predict Using Model", bg= "white" )

    caption_Lab.configure(text=category)
    predict_Lab.configure(text="Model predicts: ")
    predict_Lab.configure(bg="black", foreground = "white")




def model_predict():
    path = rand_img_path
    
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image_sized = cv2.resize(img,(64,64))
    X = np.array(image_sized).reshape(-1,64,64,1)/255
    y = int(round(model.predict(X)[0,0],0))
    prediction = CATEGORIES[y]
    predict_Lab.configure(text=f"Model predicts: {prediction}")

    if prediction == category:
        predict_Lab.configure(bg="green")
    else:
        predict_Lab.configure(bg="red")



root = Tk()
root.geometry("300x500")  
root.title("Chest X-ray prediction")
root.iconbitmap("./images/icon.ico")
root.resizable(0,0)
root.configure(background = "black")

init_img_path = ".\images\init.jpg"


root.img_init = ImageTk.PhotoImage((Image.open(init_img_path)).resize((300,300)))

image_Lab = Label(root,image = root.img_init)
image_Lab.pack()


caption_Lab = Label(root,bg= "black",foreground="white", text = "")
caption_Lab.pack()

Next_Btn = Button(root,text="Start", command = next)
Next_Btn.pack()
Next_Btn.place(x=10,y=450)

Predict_Btn = Button(root,text="", bg = "black", highlightthickness=0 ,command = model_predict)
Predict_Btn.pack()
Predict_Btn.place(x=10,y=350)

predict_Lab = Label(root, bg= "black",foreground="white" ,text = "")
predict_Lab.pack()
predict_Lab.place(x=10,y=380)




root.mainloop() 
