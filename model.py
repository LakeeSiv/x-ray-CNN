




"""
Model was first created in the mode.py file, and Keras Tuning as beggining to be implemented in this file.
However due to the large processing power required to perform Tuning, all this code was moved to and updated 
to the colabKerasTuning.ipynb file.

This file was run on Google Colab, the key benefit was that it enabled me to use GPU acceleration,
hence significantly reduced the tuning time
"""















import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from kerastuner.tuners import RandomSearch

EPOCHS = 5


X_train = np.load("./npydata/X_train.npy")
y_train = np.load("./npydata/y_train.npy")

X_test = np.load("./npydata/X_test.npy")
y_test = np.load("./npydata/y_test.npy")

#normalize data
X_train = X_train/255
X_test = X_test/255


def build_model(hp):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1), padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    for i in range(hp.Int('ConvLayers', 2, 3)):
        model.add(Conv2D(hp.Choice("filters", values = [32,64]), (3, 3), activation='relu', padding='same'))



    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation = 'sigmoid'))


    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials = 1,
    executions_per_trial=1,
    directory="./",
    project_name="tuner_models"
)

tuner.search(X_train, y_train,epochs=EPOCHS,validation_split=0.1)

best_model = tuner.get_best_models()[0]
best_mode.save("./best_model")
