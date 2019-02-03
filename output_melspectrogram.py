import matplotlib.pyplot as plt
import os
import librosa
import librosa.feature
import librosa.display
import numpy as np

path = "./music"
files = os.listdir(path)

for music_file in files:
	print(music_file+" started.")

	if not os.path.exists("./music/"+music_file+"/spectrogram"):
		os.mkdir("./music/"+music_file+"/spectrogram")

	split_data = os.listdir("./music/"+music_file+"/split_data")

	for i in range(len(split_data)):
		y, sr = librosa.load("./music/"+music_file+"/split_data/"+str(i)+".mp3")
		S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000)

		fig = plt.figure(figsize=(1, 4))
		librosa.display.specshow(librosa.power_to_db(S,ref=np.max), fmax=8000)
		plt.savefig("./music/"+music_file+"/spectrogram/"+str(i)+".jpg")
		plt.close(fig)
		fig.clf()

	print(music_file+" finished.")
