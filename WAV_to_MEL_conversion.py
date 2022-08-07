import os
import librosa.display
import numpy as np
from parameters import *

dir_str = "./DRUM_LOOPS_DS"
directory = os.fsencode(dir_str)

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    filename_wo_ext = filename[:-4]
    if filename.endswith(".wav"):
        y, SR = librosa.load("./DRUM_LOOPS_DS/" + filename, sr=SR, duration=DURATION)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=SR)
        print('Estimated tempo: {:.2f} beats per minute'.format(tempo))

        mspec = librosa.feature.melspectrogram(y=y,
                                               sr=SR,
                                               n_fft=N_FFT,
                                               hop_length=HOP_LENGTH,
                                               win_length=None,
                                               window='hann',
                                               center=True,
                                               pad_mode='reflect',
                                               power=2.0,
                                               n_mels=N_MELS)
        print(mspec.shape)
        np.save("./MEL_DRUMS_Converted/" + filename_wo_ext, mspec)
        continue
    else:
        continue
