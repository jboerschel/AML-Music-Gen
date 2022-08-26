import argparse
import os

import denorm as denorm
import librosa
import soundfile as sf
import numpy as np

from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image, make_grid
import torchvision.transforms as tt
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torch.autograd import Variable
import torch.nn.functional as F

import torch.nn as nn
import torch
from torch import autograd

from parameters import *

os.makedirs("images", exist_ok=True)
os.makedirs("audio", exist_ok=True)

cuda = True if torch.cuda.is_available() else False

dir_str = "./MEL_DRUMS_Converted"
directory = os.fsencode(dir_str)

# load data
data_list = []
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    filename_wo_ext = filename[:-4]
    if filename.endswith(".npy"):
        data_list.append(np.load(dir_str + "/" + filename)[None, :])
        continue
    else:
        continue

data = np.array(data_list)  # / max_frequency  # normalize array

num_data = data.shape[0]
pow_size = data.shape[3]
input_size = pow_size * N_MELS


# clip length to power of two (in order to use stride=2)
def find_pow_two(x):
    i = 2
    while (i < x):
        i = i * 2
    return i // 2


WIDTH = find_pow_two(pow_size)
WIDTH = 128  # TODO: Delete
data = data[:, :, :, 0:WIDTH]

d_mean = np.mean(data, axis=0)
d_std = np.std(data, axis=0)


def normalize(s):
    assert s.shape[0] == d_mean.shape[0]
    norm_Y = (s - d_mean) / (3.0 * d_std)
    return np.clip(norm_Y, -1.0, 1.0)


def denormalize(norm_s):
    assert norm_s.shape[0] == d_mean.shape[0]
    Y = (norm_s * (3.0 * d_std)) + d_mean
    return Y


for idx, d in enumerate(data):
    data[idx] = normalize(d)

DIM = 64

DS_HEIGHT = N_MELS // 2 ** 5
DS_WIDTH = WIDTH // 2 ** 5


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv_blocks = nn.Sequential(

            nn.ConvTranspose2d(LATENT_DIMS, 1024, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),

            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

        """
        self.init_size = WIDTH // 4
        # self.l1 = nn.Sequential(nn.Linear(LATENT_DIMS, DIM * 16 * DS_WIDTH * DS_HEIGHT))
        if GAN_LOSS_MODE == "wgan-gp" or GAN_LOSS_MODE == "wgan":
            self.conv_blocks = nn.Sequential(
                nn.ConvTranspose2d(LATENT_DIMS, DIM * 8, 4, stride=2, padding=1, output_padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(DIM * 8, DIM * 4, 4, stride=2, padding=1, output_padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(DIM * 4, DIM * 2, 4, stride=2, padding=1, output_padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(DIM * 2, DIM, 4, stride=2, padding=1, output_padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(DIM, 3, 4, stride=2, padding=1, output_padding=1),  # todo: SET BACK TO 1
                nn.Tanh(),
            )
        else:
            self.conv_blocks = nn.Sequential(
                # nn.BatchNorm2d(DIM * 16),
                # nn.ConvTranspose2d(DIM * 16, DIM * 8, 3, stride=2, padding=1, output_padding=1),
                nn.ConvTranspose2d(LATENT_DIMS, DIM * 8, 4, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(DIM * 8, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(DIM * 8, DIM * 4, 4, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(DIM * 4, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(DIM * 4, DIM * 2, 4, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(DIM * 2, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(DIM * 2, DIM, 4, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(DIM, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(DIM, 3, 4, stride=2, padding=1, output_padding=1),  # todo: SET BACK TO 1
                nn.Tanh(),
            )"""

    def forward(self, z):
        # out = self.l1(z)
        # out = out.view(out.shape[0], DIM * 16, DS_HEIGHT, DS_WIDTH)
        img = self.conv_blocks(z)  # out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        """        
    def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1),  # todo ksize 3
                     nn.LeakyReLU(0.2, inplace=True)]  # ,nn.Dropout2d(0.25)] -> not in SpecGAN
            if bn and not GAN_LOSS_MODE in ["wgan", "wgan-gp"]:
                block.append(nn.BatchNorm2d(out_filters))
            return block

        self.model = nn.Sequential(
            *discriminator_block(3, DIM),  # TODO : BACK TO 1
            *discriminator_block(DIM, DIM * 2),
            *discriminator_block(DIM * 2, DIM * 4),
            *discriminator_block(DIM * 4, DIM * 8),
            # *discriminator_block(DIM * 8, DIM * 16),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),  # todo remove
            nn.Flatten(),
            nn.Sigmoid()

        )

        # The height and width of downsampled image
        DS_HEIGHT = N_MELS // 2 ** 5
        DS_WIDTH = WIDTH // 2 ** 5
        if GAN_LOSS_MODE == "wgan":
            self.adv_layer = nn.Sequential(
                nn.Linear(DIM * 16 * DS_HEIGHT * DS_WIDTH, 1))  # sigmoid is not used for Wasserstein GAN
        if GAN_LOSS_MODE == "wgan-gp":
            self.adv_layer = nn.Sequential(
                nn.Linear(DIM * 16 * DS_HEIGHT * DS_WIDTH, 1))  # sigmoid is not used for Wasserstein GAN
        else:
            # todo: uncomment later
            # self.adv_layer = nn.Sequential(
            #    nn.Linear(DIM * 16 * DS_HEIGHT * DS_WIDTH, 1), nn.Sigmoid())
            pass
"""
        self.model = nn.Sequential(

            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=0, bias=False),

            nn.Flatten(),
            nn.Sigmoid()
        )

    def forward(self, img):
        out = self.model(img)
        # todo: out = out.view(out.shape[0], -1)
        # todo: validity = self.adv_layer(out)

        return out  # todo: validity


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()
if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

if LOAD_MODEL:
    generator.load_state_dict(torch.load("./GAN_G_01"))
    generator.eval()
    discriminator.load_state_dict(torch.load("./GAN_D_01"))
    discriminator.eval()
else:
    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

# Configure data loader
data_tensor = torch.from_numpy(data)  # transform to torch tensor

dataset = TensorDataset(data_tensor)  # , transform=transforms.ToTensor()) # create datset
# todo: poke


stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
dataset = ImageFolder("./poke_", transform=tt.Compose([
    tt.Resize(128),
    tt.CenterCrop(128),
    tt.ToTensor(),
    tt.Normalize(*stats),
    tt.RandomHorizontalFlip(p=0.5)
]))


def denorm(img_tensors):
    return img_tensors * stats[1][0] + stats[0][0]


def show_images(images, nmax=64, epoch=1):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imsave(f"./poke_result/{epoch}.jpg", make_grid(denorm(images.detach()[:nmax]), nrow=8).permute(1, 2, 0).numpy())


def showbatch(dl, nmax=64, epoch=1):
    for images in dl:
        show_images(images, nmax, epoch)
        break


# todo: end poke
dataloader = DataLoader(dataset,
                        batch_size=BS,  # int(2/3 * num_data),
                        shuffle=True)  # create your dataloader

# Optimizers
if GAN_LOSS_MODE == "wgan":
    # 0.0005
    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=LEARNING_RATE)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=LEARNING_RATE)
elif GAN_LOSS_MODE == "wgan-gp":
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(B1, B2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(B1, B2))
else:
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(B1, B2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(B1, B2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# ----------
#  Training
# ----------
def calculate_gradient_penalty(real_images, fake_images):
    real_images = real_images.detach()
    fake_images = fake_images.detach()

    eta = torch.FloatTensor(BS, 1, 1, 1).uniform_(0, 1)
    eta = eta.expand(BS, real_images.size(1), real_images.size(2), real_images.size(3))
    interpolated = eta * real_images + ((1 - eta) * fake_images)

    # define it to calculate gradient
    interpolated = Variable(interpolated, requires_grad=True)

    # calculate probability of interpolated examples
    prob_interpolated = discriminator(interpolated)

    # calculate gradients of probabilities with respect to examples
    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                              grad_outputs=torch.ones(
                                  prob_interpolated.size()),
                              create_graph=True, retain_graph=True)[0]

    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA_WGAN_GP
    return grad_penalty


def save_sample_to_audio(name="GAN_audio.wav"):
    z = Variable(Tensor(np.random.normal(0, 1, (1, LATENT_DIMS))))
    gen_img = generator(z)
    gen_img = gen_img.reshape(N_MELS, WIDTH).to('cpu').detach().numpy()
    gen_img = denormalize(gen_img[None, :])[0]
    audio = librosa.feature.inverse.mel_to_audio(gen_img, sr=SR, n_fft=N_FFT)
    # 22050, 44100
    sf.write(name, audio, SR, 'PCM_24')


one = torch.tensor(1, dtype=torch.float)
mone = one * -1

for epoch in range(EPOCHS):
    for i, imgs in enumerate(dataloader):
        imgs = imgs[0]
        batches_done = epoch * len(dataloader) + i

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        # z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], LATENT_DIMS))))
        z = torch.randn(BS, LATENT_DIMS, 1, 1)  # random latent tensors
        # Generate a batch of images
        gen_imgs = generator(z)
        d_out_real = discriminator(real_imgs)
        d_out_fake1 = discriminator(gen_imgs)
        d_out_fake2 = discriminator(gen_imgs.detach())  # TODO: .detach()?

        if "wgan" in GAN_LOSS_MODE and batches_done % 10 == 0:
            # Loss measures generator's ability to fool the discriminator
            if GAN_LOSS_MODE == "wgan":
                g_loss = -torch.mean(d_out_fake1)
                g_loss.backward()
            elif GAN_LOSS_MODE == "wgan-gp":
                g_loss = d_out_fake1.mean()
                g_loss.backward(mone)
                g_loss = -g_loss
        else:
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()

        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples

        if GAN_LOSS_MODE == "wgan":
            d_loss = torch.mean(d_out_fake2) - torch.mean(d_out_real)
            d_loss.backward()
        elif GAN_LOSS_MODE == "wgan-gp":
            d_loss_real = d_out_real.mean()
            d_loss_real.backward(mone)

            d_loss_fake = d_out_fake2.mean()
            d_loss_fake.backward(one)

            gradient_penalty = calculate_gradient_penalty(real_imgs.data, gen_imgs.data)
            gradient_penalty.backward()

            d_loss = d_loss_fake - d_loss_real + gradient_penalty
            wasserstein_d = d_loss_fake - d_loss_real

            '''
            d_loss = torch.mean(d_out_fake2) - torch.mean(d_out_real)
            wasserstein_d = d_loss.item()
            d_loss.backward()

            gradient_penalty = calculate_gradient_penalty(real_imgs.data, gen_imgs.data)
            gradient_penalty.backward()
            d_loss = d_loss + gradient_penalty
            '''
        else:

            # real_loss = adversarial_loss(d_out_real, valid)
            # fake_loss = adversarial_loss(d_out_fake2, fake)
            real_loss = F.binary_cross_entropy(d_out_real, valid)
            fake_loss = F.binary_cross_entropy(d_out_fake2, fake)

            d_loss = (real_loss + fake_loss)  # / 2 TODO: /2?
            d_loss.backward()

        # d_loss.backward(retain_graph=True)
        optimizer_D.step()

        if GAN_LOSS_MODE == "wgan":
            # WGAN weight clipping
            for p in discriminator.parameters():
                p.data.clamp_(-0.5, 0.5)

        if GAN_LOSS_MODE == "wgan-gp":
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [W dist: %f]"
                % (epoch, EPOCHS, i, len(dataloader), d_loss.item(), g_loss.item(), wasserstein_d.item())
            )
        else:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, EPOCHS, i, len(dataloader), d_loss.item(), g_loss.item())
            )

        if batches_done % SAVE_INTERVAL == 0:
            plot_imgs = gen_imgs.data[:25]
            # for idx, d in enumerate(plot_imgs):
            #    plot_imgs[idx] = denormalize(d)
            # mels_imgs = []
            # for img in plot_imgs:
            #    mels_imgs.append(Tensor(librosa.power_to_db(img, ref=np.max)))
            # save_image(mels_imgs, "images/%d.png" % batches_done, nrow=5, normalize=True)
            showbatch(plot_imgs, 10, batches_done)

        if batches_done % 100 == 0:
            # save_sample_to_audio("./audio/%d.wav" % batches_done)
            pass
torch.save(generator.state_dict(), "./GAN_G_01")
torch.save(discriminator.state_dict(), "./GAN_D_01")

save_sample_to_audio()
