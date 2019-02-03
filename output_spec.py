import matplotlib.pyplot as plt
import os
import librosa
import librosa.feature
import librosa.display
import numpy as np

y, sr = librosa.load("./raw/Eternal Harmony.mp3")
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000)

fig = plt.figure(figsize=(12,6))
librosa.display.specshow(librosa.power_to_db(S,ref=np.max), fmax=8000,y_axis='mel', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.savefig("spectrogram.jpg")
plt.close(fig)
fig.clf()

