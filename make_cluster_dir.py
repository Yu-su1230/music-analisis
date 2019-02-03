import os

path = "./music"
files = os.listdir(path)

for music_title in files:	
	os.mkdir(path+"/"+music_title+"/spectrogram/other")
	os.mkdir(path+"/"+music_title+"/spectrogram/Amero")
	os.mkdir(path+"/"+music_title+"/spectrogram/Bmero")
	os.mkdir(path+"/"+music_title+"/spectrogram/sabi")

