import tensorflow as tf
from keras.layers import Concatenate, Flatten
from custom_layers.Sampling import Sampling
# Function *******************************************************
# Filters the BEST digit ----------------
#
  # def function "best_digit" --------------------------------------------------
def best_digit_var_sigmoid(x_mix_filtrado_2, x_mix_orig, alpha, bias, slope,encoder,decoder,predictor):

  # alpha is the factor of zz_log_var

  # First decoded image --------------------------------------------------------------
  x_mix_filtrado_1 = (2 * x_mix_orig - x_mix_filtrado_2)                                    # Masked (Cochlear) x'2
  x_mix_filtrado_1 = (tf.clip_by_value(x_mix_filtrado_1, clip_value_min=0, clip_value_max=1))     # Filtered mix
  condition_encoder = predictor.predict(x_mix_filtrado_1, verbose=0) #* j * alfa     # con ponderado incremental
  condition_decoder_1 = condition_encoder
  
  #print(f"forma de x_mix_filtrado_1 antes de flat: {x_mix_filtrado_1.shape}") 
  
  #x_mix_filtrado_1 = tf.reshape(x_mix_filtrado_1, [x_mix_filtrado_1.shape[0], -1])

  #x_mix_filtrado_1 = Flatten()(x_mix_filtrado_1) 
  #print(f"forma de x_mix_filtrado_1 despues de flat: {x_mix_filtrado_1.shape}")
  
  #Xampliado_1 = Concatenate()([x_mix_filtrado_1, condition_encoder])
  #print(f"forma de Xampliado_1 : {Xampliado_1.shape}")
  #en.predict([img_sample,cond_sample])

  encoded_imgs = encoder.predict([x_mix_filtrado_1, condition_encoder], verbose=0)


#  z = Sampling()((encoded_imgs[0], encoded_imgs[1]))        # (z_mean, z_log_var)
  zz_log_var = encoded_imgs[1] + alpha

  #zz_log_var = tf.ones_like(encoded_imgs[1]) * (-1000000)
  z = Sampling()((encoded_imgs[0], zz_log_var))        # (z_mean, z_log_var)

  #latent_inputs = Concatenate()([z, condition_decoder_1])

  x_decoded_1 = decoder.predict([z, condition_decoder_1], verbose=0)
  x_decoded_1 = (x_decoded_1 - bias) * slope #son parametros entrenable?
  x_decoded_1 = tf.sigmoid(x_decoded_1)
  #min_value = tf.reduce_min(x_decoded_1, axis=1, keepdims=True)
  #max_value = tf.reduce_max(x_decoded_1, axis=1, keepdims=True)
  #x_decoded_1 = (x_decoded_1 - min_value) / (max_value - min_value)                                # Se satura la imagen de salida
  #x_decoded_1 = (x_decoded_1 - bias) * slope
  #x_decoded_1 = tf.sigmoid(x_decoded_1)
  x_mix_filtrado_1 = (2 * x_mix_orig * x_decoded_1)                                    # Masked (Cochlear)
  x_mix_filtrado_1 = tf.clip_by_value(x_mix_filtrado_1, clip_value_min=0, clip_value_max=1)


  return (x_mix_filtrado_1, x_decoded_1)