# -*- coding: utf-8 -*-
import os
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence
from pydub.silence import detect_nonsilent

path = "./row/namae"
files = os.listdir(path)

for music_file in files:
	music_title = os.path.splitext(os.path.basename(music_file))

	print(music_file + " started.")

	# wavファイルのデータ取得

	sound = AudioSegment.from_file("row/namae/" + music_title[0] + ".mp3", format="mp3")

	not_silence_ranges = detect_nonsilent(sound, min_silence_len=100, silence_thresh=-65)
	print(not_silence_ranges)
	chunks = []

	chunks.append(sound[not_silence_ranges[0][0]:not_silence_ranges[-1][1]])

	for i, chunk in enumerate(chunks):
		chunk.export("music/" + music_file, format="mp3", bitrate="320k")

	print(music_file + " finished.")
