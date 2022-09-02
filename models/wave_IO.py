#%%
import librosa
import librosa.display
import glob
import numpy as np
import wave

#filepath/filename*.wav -> waves, srs
def load_files_with_grob(filename):
    filenames = glob.glob(filename)
    filenames = np.array(filenames)
    filenames = np.sort(filenames)
    wvdata = np.empty((len(filenames),200000))
    srs = np.empty((len(filenames)))
    for i in range(len(filenames)):
        y, sr = librosa.load(filenames[i])
        wvdata[i] = np.pad(y, [(0,200000-len(y))])
    return wvdata, srs

#ndarray(shape:(-1,), monoral frames) -> filename(.wav)
def write_wave(filename, data, fs=48000):
    writewave = wave.Wave_write(filename)
    writewave.setparams((
        1,                      #channel
        2,                      #byte width
        fs,                     #sampling rate
        len(data),              #number of frames
        'NONE', 'not compressed'#no compression
    ))
    writewave.writeframes(array.array('h', data).tostring())
    writewave.close()


#waves, srs = load_files_with_grob('./data/JKspeech/J*.wav')
#print(waves.shape)
#print(srs.shape)
