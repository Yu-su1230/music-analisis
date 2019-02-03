
from keras.layers import Activation, Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential, load_model
from keras.utils.np_utils import to_categorical
from keras import optimizers
from keras.layers import Dense, Dropout, Flatten, Input, BatchNormalization
from keras.models import Model, Sequential
from keras.utils import plot_model
from PIL import Image
import numpy as np

from sklearn.model_selection import train_test_split
import cv2
import os

path = "./music/花ざかりWeekend/spectrogram/other"
files = os.listdir(path)
X = []
for i, spec in enumerate(files):
	Y = []
	img = cv2.imread(path+"/"+spec)
	b,g,r = cv2.split(img)
	img = cv2.merge([r,g,b])
	Y.append(img)
	Y = np.array(Y)
	X.append(Y)


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

for x in X:
	pre = model.predict(x)
	print(pre)
	print(np.argmax(pre))
