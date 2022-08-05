import os
import parameters
import librosa.display
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


x = np.array([np.array([[1,2,7],[7,8,9]]),np.array([[3,4,7],[5,6,7]])])
print(type(x[0]))
print(x.reshape(x.shape[0],x.shape[1]*x.shape[2]))


x_t = torch.Tensor(x)
print(x_t.shape)
