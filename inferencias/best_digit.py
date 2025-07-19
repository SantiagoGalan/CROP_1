import tensorflow as tf
from keras.layers import Concatenate, Flatten
from custom_layers.Sampling import Sampling
# Function *******************************************************

# def function "best_digit" --------------------------------------------------
def best_digit_var_sigmoid(x_mix_filtrado_2, x_mix_orig, alpha, bias, slope,encoder,decoder,predictor):

  # First decoded image --------------------------------------------------------------
  x_mix_filtrado_1 = (2 * x_mix_orig - x_mix_filtrado_2)                                    # Masked (Cochlear) x'2
  x_mix_filtrado_1 = (tf.clip_by_value(x_mix_filtrado_1, clip_value_min=0, clip_value_max=1))     # Filtered mix
  condition_encoder = predictor.predict(x_mix_filtrado_1, verbose=0) #* j * alfa     # con ponderado incremental
  condition_decoder_1 = condition_encoder

  encoded_imgs = encoder.predict([x_mix_filtrado_1, condition_encoder], verbose=0)

  zz_log_var = encoded_imgs[1] + alpha

  z = Sampling()((encoded_imgs[0], zz_log_var))        # (z_mean, z_log_var)

  x_decoded_1 = decoder.predict([z, condition_decoder_1], verbose=0)
  x_decoded_1 = (x_decoded_1 - bias) * slope #son parametros entrenable?
  x_decoded_1 = tf.sigmoid(x_decoded_1)

  x_mix_filtrado_1 = (2 * x_mix_orig * x_decoded_1)                                    # Masked (Cochlear)
  x_mix_filtrado_1 = tf.clip_by_value(x_mix_filtrado_1, clip_value_min=0, clip_value_max=1)

  return (x_mix_filtrado_1, x_decoded_1)