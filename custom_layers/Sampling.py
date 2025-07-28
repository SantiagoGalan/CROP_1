import tensorflow as tf
from keras.layers  import Layer

class Sampling(Layer):
  def call(self, inputs):
    z_mean, z_log_var = inputs
    batch = tf.shape(z_mean)[0]                                                 # batch = number of data in the batch
    dim = tf.shape(z_mean)[1]                                                   # dim   = number of dimensions of "z"
      # by default, random_normal has mean=0 and std=1.0
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.keras.backend.exp(0.5 * z_log_var) * epsilon

