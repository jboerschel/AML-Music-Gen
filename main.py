# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import librosa
import librosa.display
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from os import path

duration = 60
n_fft = 2048

# files
src = "On-My-Way-Lofi-Study-Music.mp3"
dst = "test.wav"

# convert wav to mp3
sound = AudioSegment.from_mp3(src)
sound.export(dst, format="wav")
name = "test.wav" # "On-My-Way-Lofi-Study-Music.wav"


# Load the audio as a waveform `y`
# Store the sampling rate as `sr`
y, sr = librosa.load(name, duration = duration)
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
print('Estimated tempo: {:.2f} beats per minute'.format(tempo))




mspec = librosa.feature.melspectrogram(y=y,
                                       sr=sr,
                                       n_fft=n_fft,
                                       hop_length=512,
                                       win_length=None,
                                       window='hann',
                                       center=True,
                                       pad_mode='reflect',
                                       power=2.0,
                                       n_mels=128)

# taking a look at the data
print(y.shape)
print(mspec.shape)
fig, ax = plt.subplots()
S_dB = librosa.power_to_db(mspec, ref=np.max)
img = librosa.display.specshow(S_dB, x_axis='time',
                               y_axis = 'mel',
                               sr = sr,
                               fmax = 8000,
                               ax = ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title='Mel-frequency spectrogram')
fig.show()


# invert mel spectogram
audio = librosa.feature.inverse.mel_to_audio(mspec, sr=sr, n_fft=n_fft)
# 22050, 44100
sf.write('inverse_mel_audio.wav', audio, sr, 'PCM_24')