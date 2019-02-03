import librosa
from pydub import AudioSegment
import os

path = "./raw"
files = os.listdir(path)

for music_file in files:
	music_title = os.path.splitext(os.path.basename(music_file))

	file_dir = "./raw/" + music_file
	y, sr = librosa.load(file_dir)
	onset_env = librosa.onset.onset_strength(y, sr=sr)
	tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
	one_bar = 1000*60*4/tempo[0]
	play_time = int(librosa.samples_to_time(len(y),sr)/(one_bar/1000))

	pre_time = 0
	rea_time = one_bar
	print(music_title[0] + " " + str(tempo[0]))

	os.mkdir("./music/"+music_title[0])
	os.mkdir("./music/"+music_title[0]+"/split_data")

	for k in range(play_time):
		audio_data = AudioSegment.from_file(file_dir, format="mp3")
		split_audio = audio_data[pre_time:rea_time]
		pre_time += one_bar
		rea_time += one_bar
		split_audio.export("./music/" + music_title[0] +"/split_data/" + str(k) + ".mp3", format = "mp3")
