import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

EPOCHS = 5


X_train = np.load("./npydata/X_train.npy")
y_train = np.load("./npydata/y_train.npy")

X_test = np.load("./npydata/X_test.npy")
y_test = np.load("./npydata/y_test.npy")

#normalize data
X_train = X_train/255
X_test = X_test/255


#<----------------Model----------------->
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1), padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation = 'sigmoid'))
#<----------------Model----------------->

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train,y_train, epochs = EPOCHS, batch_size=32,validation_split=0.1)

test_accuracy = model.evaluate(X_test,y_test, batch_size = 32)

print(f'Testing Accuracy: {(test_accuracy[1] * 100)}%')
