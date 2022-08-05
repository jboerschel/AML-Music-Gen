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

device = torch.device('mps')

dir_str = "./MEL_Converted"
directory = os.fsencode(dir_str)

# load data
data_list = []
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    filename_wo_ext = filename[:-4]
    if filename.endswith(".npy"):
        data_list.append(np.load("./MEL_Converted/" + filename)[None, :])
        continue
    else:
        continue

data = np.array(data_list)  # / max_frequency  # normalize array
print(data.shape)
print(np.max(data))
print(np.min(data))
print(np.any(np.isnan(data)))
num_data = data.shape[0]
pow_size = data.shape[3]
input_size = pow_size * N_MELS


# clip length to power of two (in order to use stride=2)
def find_pow_two(x):
    i = 2
    while (i < x):
        i = i * 2
    return i // 2


pow_size = find_pow_two(pow_size)
data = data[:, :, :, 0:pow_size]
print(data.shape)


def conv_size_out(in_size, padding, KERNEL_SIZE, stride, dilation):
    return (in_size + 2 * padding - dilation * (KERNEL_SIZE - 1) - 1) // stride + 1


out_size_h = N_MELS
out_size_w = pow_size
for i in range(N_CONV_LAYERS):
    if i in STRIDE_IND:
        out_size_h = conv_size_out(out_size_h, padding, KERNEL_SIZE, STRIDE, dilation)
        out_size_w = conv_size_out(out_size_w, padding, KERNEL_SIZE, STRIDE, dilation)
    else:
        out_size_h = conv_size_out(out_size_h, padding, KERNEL_SIZE, 1, dilation)
        out_size_w = conv_size_out(out_size_w, padding, KERNEL_SIZE, 1, dilation)
#

lin_input_size = 256 * out_size_w * out_size_h


class Encoder(nn.Module):
    def __init__(self, LATENT_DIMS):
        super(Encoder, self).__init__()
        # self.linear1 = nn.Linear(input_size, int(2 / 3 * input_size))
        # self.linear2 = nn.Linear(int(2 / 3 * input_size), LATENT_DIMS )
        self.encConv1 = nn.Conv2d(1, 16,
                                  kernel_size=KERNEL_SIZE,
                                  stride=1,
                                  padding=padding,
                                  padding_mode=padding_mode,
                                  dilation=dilation)

        self.encConv2 = nn.Conv2d(16, 32,
                                  kernel_size=KERNEL_SIZE,
                                  stride=STRIDE,
                                  padding=padding,
                                  padding_mode=padding_mode,
                                  dilation=dilation)
        self.encConv3 = nn.Conv2d(32, 64,
                                  kernel_size=KERNEL_SIZE,
                                  stride=1,
                                  padding=padding,
                                  padding_mode=padding_mode,
                                  dilation=dilation)
        self.encConv4 = nn.Conv2d(64, 128,
                                  kernel_size=KERNEL_SIZE,
                                  stride=STRIDE,
                                  padding=padding,
                                  padding_mode=padding_mode,
                                  dilation=dilation)
        self.encConv5 = nn.Conv2d(128, 256,
                                  kernel_size=KERNEL_SIZE,
                                  stride=1,
                                  padding=padding,
                                  padding_mode=padding_mode,
                                  dilation=dilation)

        self.encFC1 = nn.Linear(lin_input_size, LATENT_DIMS)
        self.encFC2 = nn.Linear(lin_input_size, LATENT_DIMS)

    def forward(self, x):
        # x = torch.flatten(x, start_dim=1)
        x = F.relu(self.encConv1(x))
        x = F.relu(self.encConv2(x))
        x = F.relu(self.encConv3(x))
        x = F.relu(self.encConv4(x))
        x = F.relu(self.encConv5(x))
        x = x.view(-1, 256 * out_size_h * out_size_w)

        mu = self.encFC1(x)
        logVar = self.encFC2(x)

        return mu, logVar


def reparameterize(mu, logVar):
    std = torch.exp(logVar / 2)
    return mu + std * torch.randn_like(std)


class Decoder(nn.Module):
    def __init__(self, LATENT_DIMS):
        super(Decoder, self).__init__()
        self.decFC1 = nn.Linear(LATENT_DIMS, lin_input_size)

        self.decConv1 = nn.ConvTranspose2d(256, 128,
                                           kernel_size=KERNEL_SIZE,
                                           stride=1,
                                           padding=padding,
                                           padding_mode=padding_mode,
                                           # output_padding=1,
                                           dilation=dilation)
        self.decConv2 = nn.ConvTranspose2d(128, 64,
                                           kernel_size=KERNEL_SIZE,
                                           stride=STRIDE,
                                           padding=padding,
                                           padding_mode=padding_mode,
                                           output_padding=1,
                                           dilation=dilation)
        self.decConv3 = nn.ConvTranspose2d(64, 32,
                                           kernel_size=KERNEL_SIZE,
                                           stride=1,
                                           padding=padding,
                                           padding_mode=padding_mode,
                                           # output_padding=1,
                                           dilation=dilation)
        self.decConv4 = nn.ConvTranspose2d(32, 16,
                                           kernel_size=KERNEL_SIZE,
                                           stride=STRIDE,
                                           padding=padding,
                                           padding_mode=padding_mode,
                                           output_padding=1,
                                           dilation=dilation)
        self.decConv5 = nn.ConvTranspose2d(16, 1,
                                           kernel_size=KERNEL_SIZE,
                                           stride=1,
                                           padding=padding,
                                           padding_mode=padding_mode,
                                           # output_padding=1,
                                           dilation=dilation)

    def forward(self, z):
        z = F.relu(self.decFC1(z))

        z = z.view(-1, 256, out_size_h, out_size_w)
        z = F.relu(self.decConv1(z))
        z = F.relu(self.decConv2(z))
        z = F.relu(self.decConv3(z))
        z = F.relu(self.decConv4(z))
        z = torch.sigmoid(self.decConv5(z))
        return z  # .reshape((-1, 1, N_MELS, pow_size))


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
    opt = torch.optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)
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

            print(x_hat.shape)
            print(torch.max(x_hat))
            print(torch.min(x_hat))

            kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
            print(kl_divergence)
            # print(F.binary_cross_entropy(x_hat, x, size_average=False))
            # loss_cross_entropy = F.binary_cross_entropy(x_hat, x, size_average=False)
            loss_mse = ((x - x_hat) ** 2).mean()
            print(loss_mse)
            loss = kl_divergence + loss_mse
            print("Loss = {}".format(loss))
            loss_l.append(loss.item())

            loss.backward()
            opt.step()
        print("Average Loss at Epoch {epoch}: {loss}".format(epoch=epoch, loss=sum(loss_l) / len(loss_l)))
        if epoch % 10 == 0:
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
