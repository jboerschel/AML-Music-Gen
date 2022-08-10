# mel spectogram parameters
N_MELS = 128
LENGTH = 256
DURATION = 6
N_FFT = 2048
HOP_LENGTH = 512
SR = 22050
max_frequency = SR / 2  # nyquist frequency

# model parameters
LOAD_MODEL = False
GAN_LOSS_MODE = "wgan-gp"  # wgan, wgan-gp, anything else results in standard GAN
LAMBDA_WGAN_GP = 10
# adam
LEARNING_RATE = 1e-4
B1 = 0.5
B2 = 0.9
DESCENDING_RATE = True
DESCENDING_RATE_TAU = 0.9
# other
EPOCHS = 10000
LATENT_DIMS = 100
BS = 60
N_CONV_LAYERS = 5
KERNEL_SIZE = 5
STRIDE = 2
STRIDE_IND = [1, 3]
padding = int(KERNEL_SIZE / 2)
padding_mode = 'zeros'
dilation = 1

# general param

SAVE_INTERVAL = 20
