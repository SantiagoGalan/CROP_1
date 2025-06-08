import numpy as np
import tensorflow as tf

## Compile Loss and Optimizer

#  # Loss -----------------------------------------------------------------------
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
  """
  Función de pérdida personalizada para un VAE condicional.

  Parámetros:
  -----------
  y_true : tensor
      Imagen original (ground truth).
  y_pred : lista de tensores
      [reconstrucción, z_mean, z_log_var]

  Retorna:
  --------
  loss : tensor
      Pérdida total (reconstrucción + divergencia KL).
  """

  recon = y_pred[0]        # Imagen reconstruida por el decoder
  z_mean = y_pred[1]       # Media del espacio latente
  z_log_var = y_pred[2]    # Log-varianza del espacio latente

  # Cálculo de la divergencia KL entre la distribución latente y la normal estándar
  kl_loss = -0.5 * (z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
  kl_loss = tf.reduce_sum(kl_loss, -1)
  
  # Cálculo del error cuadrático medio (MSE) entre la imagen original y la reconstruida
  mse = tf.keras.losses.MeanSquaredError()
  mse1 = mse(y_true, recon) * tf.cast(tf.reduce_prod(tf.shape(y_true)[1:]), tf.float32)  # Número de píxeles dinámico

  # Suma de la reconstrucción y la divergencia KL (Beta=1)
  loss = mse1 + 1. * kl_loss
  
  return tf.reduce_mean(loss) #, axis=-1)