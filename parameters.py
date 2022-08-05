# mel spectogram parameters
N_MELS = 128
DURATION = 5
N_FFT = 2048
HOP_LENGTH = 512
SR = 22050
max_frequency = SR / 2  # nyquist frequency

# model parameters
LEARNING_RATE = 0.0001
EPOCHS = 2
LATENT_DIMS = 10
BS = 3
N_CONV_LAYERS = 5
KERNEL_SIZE = 5
STRIDE = 2
STRIDE_IND = [1, 3]
padding = int(KERNEL_SIZE / 2)
padding_mode = 'zeros'
dilation = 1
