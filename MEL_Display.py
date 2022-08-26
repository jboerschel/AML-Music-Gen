import os
from parameters import *
import librosa.display
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

from parameters import *

dir_str = "./MEL_DRUMS_Converted"
directory = os.fsencode(dir_str)
n_mels = N_MELS
duration = DURATION
n_fft = N_FFT
hop_length = HOP_LENGTH
sr = SR

latent_dims = LATENT_DIMS
bs = BS
# load data
data_list = []
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    filename_wo_ext = filename[:-4]
    if filename.endswith(".npy"):
        data_list.append(np.load(dir_str+"/"+filename))
        continue
    else:
        continue

data = np.array(data_list)
print(data.shape)
num_data = data.shape[0]
pow_size = data.shape[2]
input_size = pow_size * n_mels

for i in range(3):
    mspec = data[i]
    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(mspec, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time',
                                   y_axis='mel',
                                   sr=SR,
                                   fmax=8000,
                                   ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    fig.show()