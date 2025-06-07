import numpy as np
import tensorflow as tf
from keras.layers import  Concatenate
from keras.datasets import mnist
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy

from keras import backend as K
import matplotlib as mplt
import matplotlib.pyplot as plt

## Compile Loss and Optimizer

  # Loss -----------------------------------------------------------------------
'''
def loss(y_true, y_pred):

  recon = y_pred[0]
  z_mean = y_pred[1]
  z_log_var = y_pred[2]

  binary_crossentropy = BinaryCrossentropy()
  
  Beta = 1.                                                          # Beta-VAE
  #z_mean = y_pred[0]
  #z_log_var = y_pred[1]
  #z = y_pred[2]

  kl_loss = -0.5 * (z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
  kl_loss = tf.reduce_sum(kl_loss, -1)

  mse = tf.keras.losses.MeanSquaredError()
  mse1 = mse(y_true, y_pred) * original_dim

  # Add KL divergence regularization loss.
  loss = mse1 + Beta * kl_loss
  return tf.reduce_mean(loss) #, axis=-1)
'''
def vae_loss(y_true, y_pred):
  recon = y_pred[0]
  z_mean = y_pred[1]
  z_log_var = y_pred[2]
  kl_loss = -0.5 * (z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
  kl_loss = tf.reduce_sum(kl_loss, -1)
  
  mse = tf.keras.losses.MeanSquaredError()
  mse1 = mse(y_true, y_pred) * 794 #TODO
  
  # Add KL divergence regularization loss.
  loss = mse1 + 1. * kl_loss
  
  return tf.reduce_mean(loss) #, axis=-1)
