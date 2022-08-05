import os
import librosa.display
import numpy as np
import parameters

dir_str = "./WAV_Converted"
directory = os.fsencode(dir_str)
n_mels = parameters.n_mels
duration = parameters.duration
n_fft = parameters.n_fft
hop_length = parameters.hop_length
sr=parameters.sr


for file in os.listdir(directory):
    filename = os.fsdecode(file)
    filename_wo_ext = filename[:-4]
    if filename.endswith(".wav"):
        y, sr = librosa.load("./WAV_Converted/"+filename, sr=sr, duration=duration)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        #print('Estimated tempo: {:.2f} beats per minute'.format(tempo))

        mspec = librosa.feature.melspectrogram(y=y,
                                               sr=sr,
                                               n_fft=n_fft,
                                               hop_length=hop_length,
                                               win_length=None,
                                               window='hann',
                                               center=True,
                                               pad_mode='reflect',
                                               power=2.0,
                                               n_mels=n_mels)
        print(mspec.shape)
        np.save("./MEL_Converted/" + filename_wo_ext, mspec)
        continue
    else:
        continue