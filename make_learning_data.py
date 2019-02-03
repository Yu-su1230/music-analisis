from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import cv2
import os
import numpy as np

path = "./music/"
categories = ["Amero", "Bmero", "other", "sabi"]
music_list = ["Be mine!", "Behind Moon", "Eternal Harmony", "Melty Fantasia", "STARTLINER", "Snow in “I love you”", "ambiguous","dear...", "motto☆派手にね！", "花ざかりWeekend"]

X = []
y = []

for music_title in music_list:
	print(music_title)

	for idx, cat in enumerate(categories):
		print(cat)
		print(idx)
		spr_dir = path+music_title+"/spectrogram/"+cat
		files = os.listdir(spr_dir)
		for spr_file in files:
			img = cv2.imread(spr_dir + "/" + spr_file)
			b,g,r = cv2.split(img)
			img = cv2.merge([r,g,b])
			X.append(img)
			y.append(idx)

X = np.array(X)
y = np.array(y)
y = y.T
y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X,y)
xy = (X_train,X_test,y_train, y_test)
np.save("./spectrogram.npy", xy)

