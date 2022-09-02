#%%
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display



def show_wave(y):
    plt.plot(y, label="wave")
    plt.legend()
    plt.show()

def show_spec(y, sr):
    D = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    plt.figure(figsize=(16, 6))
    plt.plot(D)
    plt.grid()
    plt.show()

    DB = librosa.amplitude_to_db(D, ref=np.max)
    plt.figure(figsize=(16, 6))
    librosa.display.specshow(DB, sr=sr, hop_length=512, x_axis='time', y_axis='log')
    plt.colorbar()
    plt.show()

def show_melspec(y, sr):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, win_length=512, hop_length=512)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time',y_axis='mel', sr=sr)
    plt.colorbar(img, format='%+2.0f dB')
    plt.show()
