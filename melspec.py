import matplotlib.pyplot as plt
import librosa
import librosa.feature
import librosa.display
import numpy as np

music_title = "vocal/Fa.mp3"

n_mels=128
hop_length=2068
n_fft=2048

y, sr = librosa.load(music_title)

S = librosa.feature.melspectrogram(y=y,sr=sr,n_mels=n_mels,hop_length=hop_length,n_fft=n_fft)
log_S = librosa.power_to_db(S,ref=np.max)

plt.figure(figsize=(12, 4))
librosa.display.specshow(data=log_S,sr=sr,hop_length=hop_length,y_axis='mel',x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()
plt.show()

