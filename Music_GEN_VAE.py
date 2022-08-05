import os
import parameters
import librosa.display
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms, utils
import datetime

torch.manual_seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dir_str = "./MEL_Converted"
directory = os.fsencode(dir_str)
n_mels = parameters.n_mels
duration = parameters.duration
n_fft = parameters.n_fft
hop_length = parameters.hop_length
sr = parameters.sr

latent_dims = parameters.latent_dims
bs = parameters.bs
# load data
data_list = []
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    filename_wo_ext = filename[:-4]
    if filename.endswith(".npy"):
        data_list.append(np.load("./MEL_Converted/" + filename))
        continue
    else:
        continue

data = np.array(data_list)
print(data.shape)
num_data = data.shape[0]
pow_size = data.shape[2]
input_size = pow_size * n_mels


class Encoder(nn.Module):
    def __init__(self, latent_dims):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(input_size, int(2 / 3 * input_size))
        self.linear2 = nn.Linear(int(2 / 3 * input_size), latent_dims)

    def forward(self, x):
        # x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        return self.linear2(x)


class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, int(2 / 3 * input_size))
        self.linear2 = nn.Linear(int(2 / 3 * input_size), input_size)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z  # .reshape((-1, 1, n_mels, pow_size))


class Autoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def train(autoencoder, data, epochs=20):
    opt = torch.optim.Adam(autoencoder.parameters())
    starttime = datetime.datetime.now()
    print("Starting training at {}".format(starttime))
    current_time = None
    for epoch in range(epochs):
        print("Current epoch: {}, time taken for last epoch: {}".format(epoch, (
                datetime.datetime.now() - current_time) if current_time is not None else 0))
        current_time = datetime.datetime.now()

        loss_l = []
        for x in data:
            x = x[0]
            x = x.to(device)  # GPU
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = ((x - x_hat) ** 2).sum()
            loss_l.append(loss.item())
            loss.backward()
            opt.step()
        print("Average Loss at Epoch {epoch}: {loss}".format(epoch=epoch, loss=sum(loss_l) / len(loss_l)))
    print("Finished training at {}, total time needed: {}".format(datetime.datetime.now(),
                                                                  datetime.datetime.now() - starttime))

    return autoencoder


autoencoder = Autoencoder(latent_dims).to(device)


def to_tensor(x):
    l = []
    for i, d in enumerate(x):
        l.append(torch.Tensor(d))

    return torch.Tensor(l)


data_reshape = data.reshape(data.shape[0], data.shape[1] * data.shape[2])
print(type(data_reshape[0]))
print(data_reshape[0])

# data_tensor = to_tensor(data_reshape)

data_tensor = torch.from_numpy(data_reshape)  # transform to torch tensor
dataset = TensorDataset(data_tensor)  # , transform=transforms.ToTensor()) # create datset

dataloader = DataLoader(dataset,
                        batch_size=bs,  # int(2/3 * num_data),
                        shuffle=True)  # create your dataloader

autoencoder = train(autoencoder, dataloader, epochs=50)

x = torch.randn(1, latent_dims)

z = torch.Tensor(x).to(device)
x_hat = autoencoder.decoder(z)
x_hat = x_hat.reshape(n_mels, pow_size).to('cpu').detach().numpy()

np.save("./result", x_hat)

fig, ax = plt.subplots()
S_dB = librosa.power_to_db(x_hat, ref=np.max)
img = librosa.display.specshow(S_dB, x_axis='time',
                               y_axis='mel',
                               sr=sr,
                               fmax=8000,
                               ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title='Mel-frequency spectrogram')
fig.show()
fig.savefig("./pimmel.png")

torch.save(autoencoder.state_dict(), "./Model_01")

# invert mel spectogram
audio = librosa.feature.inverse.mel_to_audio(x_hat, sr=sr, n_fft=n_fft)
# 22050, 44100
sf.write('inverse_mel_audio.wav', audio, sr, 'PCM_24')
