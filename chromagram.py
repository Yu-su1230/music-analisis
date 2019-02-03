import librosa
import librosa.display
import matplotlib.pyplot as plt

y, sr = librosa.load('vocal/Fa.mp3', sr=44100)

y_harmonic, y_percussive = librosa.effects.hpss(y)

C = librosa.feature.chroma_cens(y=y_harmonic, sr=sr)

plt.figure(figsize=(12,4))
librosa.display.specshow(C,sr=sr,x_axis='time',y_axis='chroma')
plt.colorbar()
plt.title('Chromagram')
plt.tight_layout()
plt.show()
