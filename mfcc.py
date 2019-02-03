import librosa
import librosa.display
import matplotlib.pyplot as plt

y, sr = librosa.load('music/GO! GO! ラブリズム.m4a', sr=44100)
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

plt.figure(figsize=(10,4))
librosa.display.specshow(mfccs,x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()

