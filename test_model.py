from keras.layers import Activation, Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential, load_model
from keras.utils.np_utils import to_categorical
from keras import optimizers
from keras.layers import Dense, Dropout, Flatten, Input, BatchNormalization
from keras.models import Model, Sequential
from keras.utils import plot_model
import numpy as np

from sklearn.model_selection import train_test_split
import cv2
import os

X_train, X_test, y_train, y_test = np.load("./spectrogram.npy")

print(X_test)
print(y_test)

model = Sequential()
model.add(Conv2D(input_shape=(307, 76, 3), filters=32, kernel_size=(2, 2), strides=(1, 1), padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=32, kernel_size=(2,2), strides=(1,1), padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=32, kernel_size=(2,2), strides=(1,1), padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(4))
model.add(Activation('softmax'))
model.compile(optimizer='Adam', loss='categorical_crossentropy',metrics=['accuracy'])

model.load_weights("./music_model.hdf5")

#plot_model(model, to_file="model.png")

score = model.evaluate(X_test,y_test,verbose=1)
print('loss=', score[0])
print('accuracy', score[1])
