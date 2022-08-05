import os
from parameters import *
import librosa.display
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment

import datetime
import glob
import time
import tensorflow as tf
#from tensorflow.keras import layers
from tensorflow import keras


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

layersize_1 = (input_size // 3) * 2
layersize_2 = (layersize_1 // 3) * 2
layersize_3 = (layersize_2 // 3) * 2


train_dataset = tf.convert_to_tensor(data, dtype=tf.int64)

def encoder(input_encoder):
    inputs = keras.Input(shape=input_encoder, name='input_layer')

    flatten = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(layersize_1, name='linear1')(flatten)
    x = tf.keras.layers.BatchNormalization(name='bn_1')(x)
    x = tf.keras.layers.LeakyReLU(name='lrelu_1')(x)

    x = tf.keras.layers.Dense(layersize_2, name='linear2')(x)
    x = tf.keras.layers.BatchNormalization(name='bn_2')(x)
    x = tf.keras.layers.LeakyReLU(name='lrelu_2')(x)

    x = tf.keras.layers.Dense(layersize_3, name='linear3')(x)
    x = tf.keras.layers.BatchNormalization(name='bn_3')(x)
    x = tf.keras.layers.LeakyReLU(name='lrelu_3')(x)

    mean = tf.keras.layers.Dense(LATENT_DIMS, name='mean')(x)
    log_var = tf.keras.layers.Dense(LATENT_DIMS, name='log_var')(x)
    model = tf.keras.Model(inputs, (mean, log_var), name="Encoder")
    return model


def sampling(input_1, input_2):
    mean = keras.Input(shape=input_1, name='input_layer1')
    log_var = keras.Input(shape=input_2, name='input_layer2')
    out = tf.keras.layers.Lambda(sampling_reparameterization, name='encoder_output')([mean, log_var])
    enc_2 = tf.keras.Model([mean,log_var], out,  name="Encoder_2")
    return enc_2

def sampling_reparameterization(distribution_params):
    mean, log_var = distribution_params
    epsilon = K.random_normal(shape=K.shape(mean), mean=0., stddev=1.)
    z = mean + K.exp(log_var / 2) * epsilon
    return z


def decoder(input_decoder):
    inputs = keras.Input(shape=input_decoder, name='input_layer')

    x = tf.keras.layers.Dense(layersize_3, name='linear_1')(inputs)
    x = tf.keras.layers.BatchNormalization(name='bn_1')(x)
    x = tf.keras.layers.LeakyReLU(name='lrelu_1')(x)

    x = tf.keras.layers.Dense(layersize_2, name='linear_2')(x)
    x = tf.keras.layers.BatchNormalization(name='bn_2')(x)
    x = tf.keras.layers.LeakyReLU(name='lrelu_2')(x)

    x = tf.keras.layers.Dense(layersize_1, name='linear_3')(x)
    x = tf.keras.layers.BatchNormalization(name='bn_3')(x)
    x = tf.keras.layers.LeakyReLU(name='lrelu_3')(x)

    x = tf.keras.layers.Dense(input_size, name='linear4', activation='sigmoid')(x)
    outputs = tf.keras.layers.Reshape((N_MELS,pow_size), name='Reshape_Layer')(x)
    model = tf.keras.Model(inputs, outputs, name="Decoder")
    return model




optimizer = tf.keras.optimizers.Adam(lr = LEARNING_RATE)

def mse_loss(y_true, y_pred):
    r_loss = keras.mean(keras.square(y_true - y_pred), axis = [1,2,3])
    return 1000 * r_loss

def kl_loss(mean, log_var):
    kl_loss =  -0.5 * keras.sum(1 + log_var - keras.square(mean) - keras.exp(log_var), axis = 1)
    return kl_loss

def vae_loss(y_true, y_pred, mean, log_var):
    r_loss = mse_loss(y_true, y_pred)
    kl_l = kl_loss(mean, log_var)
    return  r_loss + kl_l


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    with tf.GradientTape() as encoder, tf.GradientTape() as decoder:
        mean, log_var = enc(images, training=True)
        latent = sampling([mean, log_var])
        generated_images = dec(latent, training=True)
        loss = vae_loss(images, generated_images, mean, log_var)

    gradients_of_enc = encoder.gradient(loss, enc.trainable_variables)
    gradients_of_dec = decoder.gradient(loss, dec.trainable_variables)

    optimizer.apply_gradients(zip(gradients_of_enc, enc.trainable_variables))
    optimizer.apply_gradients(zip(gradients_of_dec, dec.trainable_variables))
    return loss

def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()
    for batch in dataset:
      train_step(batch)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))


train(train_dataset, EPOCHS)
