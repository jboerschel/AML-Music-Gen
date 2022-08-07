import os
from parameters import *
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

device = torch.device("cpu")  # mps for m1 gpu, cpu or cuda for nvidia graphics cards

dir_str = "./MEL_Converted"
directory = os.fsencode(dir_str)

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

data = np.array(data_list)  # / max_frequency  # normalize array
print(data.shape)
print(np.max(data))
print(np.min(data))
print(np.any(np.isnan(data)))
num_data = data.shape[0]
pow_size = data.shape[2]
input_size = pow_size * N_MELS

layer2_size = int(2 / 3 * input_size)
layer3_size = int(2 / 3 * layer2_size)
# normalizing
d_min = data.min(axis=(1, 2), keepdims=True)
d_max = data.max(axis=(1, 2), keepdims=True)


# data = (data - d_min) / (d_max - d_min)


class Encoder(nn.Module):
    def __init__(self, LATENT_DIMS):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(input_size, layer2_size)
        self.linear2 = nn.Linear(layer2_size, layer3_size)

        self.encFC1 = nn.Linear(layer3_size, LATENT_DIMS)
        self.encFC2 = nn.Linear(layer3_size, LATENT_DIMS)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))

        mu = self.encFC1(x)
        logVar = self.encFC2(x)

        return mu, logVar


class Decoder(nn.Module):
    def __init__(self, LATENT_DIMS):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(LATENT_DIMS, layer3_size)
        self.linear2 = nn.Linear(layer3_size, layer2_size)
        self.linear3 = nn.Linear(layer2_size, input_size)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = F.relu(self.linear2(z))
        z = F.relu(self.linear3(z))  # torch.sigmoid(self.linear3(z))
        return z.reshape((-1, 1, N_MELS, pow_size))


def reparameterize(mu, logVar):
    std = torch.exp(logVar / 2)
    return mu + std * torch.randn_like(std)


class Autoencoder(nn.Module):
    def __init__(self, LATENT_DIMS):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(LATENT_DIMS)
        self.decoder = Decoder(LATENT_DIMS)

    def forward(self, x):
        mu, logVar = self.encoder(x)
        z = reparameterize(mu, logVar)
        return self.decoder(z), mu, logVar


def train(autoencoder, data_loader, epochs=20):
    learning_rate = LEARNING_RATE

    opt = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
    starttime = datetime.datetime.now()
    print("Starting training at {}".format(starttime))
    current_time = None
    for epoch in range(epochs):
        print("Current epoch: {}, time taken for last epoch: {}".format(epoch, (
                datetime.datetime.now() - current_time) if current_time is not None else 0))
        current_time = datetime.datetime.now()

        loss_l = []

        for _, data in enumerate(data_loader, 0):
            x = data[0]
            # x = x[None, :]
            x = x.to(device)  # GPU

            opt.zero_grad()

            x_hat, mu, logVar = autoencoder(x)
            # x_hat = x_hat * max_frequency
            print(x_hat.shape)
            print(torch.max(x_hat))
            print(torch.min(x_hat))

            kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp()) / x.shape[0]
            print(kl_divergence)
            # print(F.binary_cross_entropy(x_hat, x, size_average=False))
            # loss_cross_entropy = F.binary_cross_entropy(x_hat, x, size_average=False)
            loss_mse = 0.5 * torch.sum((x - x_hat) ** 2)
            loss_mse = loss_mse.mean()
            print(loss_mse)
            loss = kl_divergence + loss_mse
            print("Loss = {}".format(loss))
            loss_l.append(loss.item())

            loss.backward()
            opt.step()
        print("Average Loss at Epoch {epoch}: {loss}".format(epoch=epoch, loss=sum(loss_l) / len(loss_l)))
        if epoch % 10 == 0:
            if DESCENDING_RATE:
                learning_rate = learning_rate * DESCENDING_RATE_TAU

            sample = torch.randn(1, LATENT_DIMS)
            z = torch.Tensor(sample).to(device)
            s = autoencoder.decoder(z)
            s = s.reshape(N_MELS, pow_size).to('cpu').detach().numpy()  # * max_frequency

            np.save("./result", s)

            fig, ax = plt.subplots()
            S_dB = librosa.power_to_db(s, ref=np.max)
            img = librosa.display.specshow(S_dB, x_axis='time',
                                           y_axis='mel',
                                           sr=SR,
                                           fmax=8000,
                                           ax=ax)
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            ax.set(title='Mel-frequency spectrogram')
            fig.show()
            fig.savefig("./pimmel.png")
    print("Finished training at {}, total time needed: {}".format(datetime.datetime.now(),
                                                                  datetime.datetime.now() - starttime))

    return autoencoder


autoencoder = Autoencoder(LATENT_DIMS).to(device)

if LOAD_MODEL:
    autoencoder.load_state_dict(torch.load("./Model_01"))
    autoencoder.eval()


def to_tensor(x):
    l = []
    for i, d in enumerate(x):
        l.append(torch.Tensor(d))

    return torch.Tensor(l)


# data_reshape = data.reshape(data.shape[0], data.shape[1] * data.shape[2])
# print(type(data_reshape[0]))
# print(data_reshape[0])

# data_tensor = to_tensor(data_reshape)

data_tensor = torch.from_numpy(data)  # transform to torch tensor

dataset = TensorDataset(data_tensor)  # , transform=transforms.ToTensor()) # create datset

dataloader = DataLoader(dataset,
                        batch_size=BS,  # int(2/3 * num_data),
                        shuffle=True)  # create your dataloader

autoencoder = train(autoencoder, dataloader, epochs=EPOCHS)

x = torch.randn(1, LATENT_DIMS)

z = torch.Tensor(x).to(device)
x_hat = autoencoder.decoder(z)
x_hat = x_hat.reshape(N_MELS, pow_size).to('cpu').detach().numpy()  # * max_frequency

np.save("./result", x_hat)

fig, ax = plt.subplots()
S_dB = librosa.power_to_db(x_hat, ref=np.max)
img = librosa.display.specshow(S_dB, x_axis='time',
                               y_axis='mel',
                               sr=SR,
                               fmax=8000,
                               ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title='Mel-frequency spectrogram')
fig.show()
fig.savefig("./pimmel.png")

torch.save(autoencoder.state_dict(), "./Model_01")

# invert mel spectogram
audio = librosa.feature.inverse.mel_to_audio(x_hat, sr=SR, n_fft=N_FFT)
# 22050, 44100
sf.write('inverse_mel_audio.wav', audio, SR, 'PCM_24')
