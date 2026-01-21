import tensorflow as tf
from keras.layers  import Layer
SEED = 1234
class Sampling(Layer):
  def call(self, inputs):
    z_mean, z_log_var = inputs
    batch = tf.shape(z_mean)[0]                                                 
    dim = tf.shape(z_mean)[1]                                                
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    #epsilon = tf.random.stateless_normal(shape=tf.shape(z_mean),seed=[SEED, 0]) # esto y la semilla anula el sampleo aleatorio. 
    return z_mean + tf.keras.backend.exp(0.5 * z_log_var) * epsilon

