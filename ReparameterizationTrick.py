import numpy as np
import tensorflow as tf
from keras import layers
from keras.layers import  Concatenate
from keras.datasets import mnist
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
mse = MeanSquaredError()
binary_crossentropy = BinaryCrossentropy()
from keras import backend as K
import matplotlib as mplt
import matplotlib.pyplot as plt
from keras.layers import Lambda, Input, Dense, Concatenate

## Reparameterization trick ----------------------------------------------------
"""                   -- Sampling --

Reparameterization trick by sampling from an isotropic unit Gaussian.
# instead of sampling from q(z|x), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps

# Arguments:
    args (tensor): mean and log of variance of q(z|x)
    Uses (z_mean, z_log_var) to sample z, the vector encoding the input "x".

# Returns:
    z (tensor): sampled latent vector
"""

class Sampling(layers.Layer):
  def call(self, inputs):
    z_mean, z_log_var = inputs
    batch = tf.shape(z_mean)[0]                                                 # batch = number of data in the batch
    dim = tf.shape(z_mean)[1]                                                   # dim   = number of dimensions of "z"
      # by default, random_normal has mean=0 and std=1.0
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.keras.backend.exp(0.5 * z_log_var) * epsilon