from PIL import Image
import os

path ="./music"
files = os.listdir(path)

for music_folder in files:
	spectrogram = os.listdir(path+"/"+music_folder+"/spectrogram")
	for i in range(len(spectrogram)):
		im = Image.open(path+"/"+music_folder+"/spectrogram/"+str(i)+".jpg")
		crop = im.crop((14,49,90,356))
		crop.save(path+"/"+music_folder+"/spectrogram/"+"crop"+str(i)+".jpg",quality=100)

