from tensorflow.keras.layers import BatchNormalization, Dropout

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.layers import Lambda, Input, Dense, Concatenate
from keras.models import Model                                                  # Ok: se usa 230201
from keras.datasets import mnist
#from keras.losses import mse, binary_crossentropy

from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
mse = MeanSquaredError()
binary_crossentropy = BinaryCrossentropy()

from keras.utils import plot_model
from keras import backend as K
from keras import optimizers
from keras import layers
import tensorflow as tf
import numpy as np
import matplotlib as mplt
import matplotlib.pyplot as plt
import argparse
import os
import pickle
import outcomes

import importlib
importlib.reload(outcomes)




def best_digit_var_sigmoid(x_mix_filtrado_2, x_mix_orig, alpha, bias, slope,predictor,encoder,decoder):

    # alpha is the factor of zz_log_var

    # First decoded image --------------------------------------------------------------
    x_mix_filtrado_1 = (2 * x_mix_orig - x_mix_filtrado_2)     # Masked (Cochlear) x'2
    
    x_mix_filtrado_1 = (tf.clip_by_value(x_mix_filtrado_1, clip_value_min=0, clip_value_max=1))     # Filtered mix


    # Si x_mix_filtrado_1 tiene shape (28, 28)
    if x_mix_filtrado_1.ndim == 2:
        x_mix_filtrado_1 = np.expand_dims(x_mix_filtrado_1, axis=0)  # (1, 28, 28)
        #x_mix_filtrado_1 = np.expand_dims(x_mix_filtrado_1, axis=-1) # (1, 28, 28, 1)

    print(x_mix_filtrado_1.shape)
    
    condition_encoder = predictor.predict(x_mix_filtrado_1)#, verbose=0 #* j * alfa     # con ponderado incremental
    condition_decoder_1 = condition_encoder
    #Xampliado_1 = tf.concat([x_mix_filtrado_1, condition_encoder], axis=-1)
    #Xampliado_1 = Concatenate()([x_mix_filtrado_1, condition_encoder])
#    encoded_imgs = encoder.predict([x_mix_filtrado_1, condition_encoder], verbose=0)
    latent_inputs = encoder.predict([x_mix_filtrado_1, condition_encoder], verbose=0)


    #  z = Sampling()((encoded_imgs[0], encoded_imgs[1]))        # (z_mean, z_log_var)
    #zz_log_var = encoded_imgs[1] + alpha

    #zz_log_var = tf.ones_like(encoded_imgs[1]) * (-1000000)
    #z = Sampling()((encoded_imgs[0], zz_log_var))        # (z_mean, z_log_var)

    #latent_inputs = Concatenate()([z, condition_decoder_1])
    #latent_inputs = tf.concat([z, condition_decoder_1],axis=-1)
    x_decoded_1 = decoder.predict([latent_inputs[2], condition_decoder_1], verbose=0)


    x_decoded_1 = (x_decoded_1 - bias) * slope  #son parametros entrenable? 
    x_decoded_1 = tf.sigmoid(x_decoded_1)
    x_decoded_1 = np.squeeze(x_decoded_1)  # Si quieres quitar dimensiones de 1
    #min_value = tf.reduce_min(x_decoded_1, axis=1, keepdims=True)
    #max_value = tf.reduce_max(x_decoded_1, axis=1, keepdims=True)
    #x_decoded_1 = (x_decoded_1 - min_value) / (max_value - min_value)                                # Se satura la imagen de salida
    #x_decoded_1 = (x_decoded_1 - bias) * slope
    #x_decoded_1 = tf.sigmoid(x_decoded_1)
    x_mix_filtrado_1 = (2 * x_mix_orig * x_decoded_1)                                    # Masked (Cochlear)
    x_mix_filtrado_1 = tf.clip_by_value(x_mix_filtrado_1, clip_value_min=0, clip_value_max=1)

    #print(x_decoded_1.type())

    return (x_mix_filtrado_1, x_decoded_1)

#-------------------------------versión 3 (TRAIN a simplificar CORTO PARA PRUEBAS CON VARIANZA)-----------------------------------------
#                     REVISAR este help
# 11 minutos
# ------------------------------------------------------------------------------
# Encode and decode some digits

## Superimposed digits - MAX
#maximum_image = np.maximum(x_train,x_train_1)
# x_train_mix = maximum_image                                                    # Habiltar para MAX - Inabilitar para AVERAGE

## Superimposed digits - AVERAGE
def inferncia_modelo(x_train,x_train_1,y_train,predictor,encoder,decoder, y_train_1):
        
    alfa_mix = 0.5
    average_image = alfa_mix * x_train.astype(np.float32) + (1 - alfa_mix) * x_train_1.astype(np.float32)
    x_train_mix = average_image                                                      # Inhabiltar para MAX - Habilitar para AVERAGE

    ## Initialization
    x_train_mix_orig = x_train_mix
    x_train_decoded_1 = x_train_mix                                                          # Added in order to improve the prediction in each iteration
    x_train_decoded_2 = x_train_mix                                                          # Added in order to improve the prediction in each iteration
    x_train_mix_IN = x_train_mix                                                        # Added in order to improve more the prediction in each iteration
    x_train_mix_filtrado_1 = x_train_mix                                                # Added in order to improve more the prediction in each iteration
    x_train_mix_filtrado_2 = x_train_mix                                                # Added in order to improve more the prediction in each iteration
    x__x = tf.zeros_like(x_train_mix)
    condition_encoder = tf.zeros_like(y_train)


    Iterations = 30

    bias = 0.22
    slope = 22.

    beta = 1.
    alpha_1 = -2
    alpha_2 = -22


    for j in range(Iterations):

        # best_digit_var_sigmoid(x_mix_filtrado_2, x_mix_orig, alpha, bias, slope)

        x_train_mix_filtrado_1, x_train_decoded_1 = best_digit_var_sigmoid(x_train_mix_filtrado_2, x_train_mix_orig, alpha_2, bias, slope,predictor,encoder,decoder)
        alpha_2 = alpha_2 * beta

        x_train_mix_filtrado_2, x_train_decoded_2 = best_digit_var_sigmoid(x_train_mix_filtrado_1, x_train_mix_orig, alpha_1, bias, slope,predictor,encoder,decoder)
        alpha_1 = alpha_1 * beta

        ######################################################

        print("ITERACIÓN A: ", j)

        #x_train_best_predicted_1, y_train_predicted_1_f, y_train_predicted_2_f = outcomes.outcomes(x_train_decoded_1, x_train_decoded_2, x_train_mix_filtrado_1, x_train_mix_filtrado_2, x_train_mix_orig, x_train, x_train_1, y_train, y_train_1,predictor)
        _, y_train_predicted_1_f, y_train_predicted_2_f = outcomes.outcomes(x_train_decoded_1, x_train_decoded_2, x_train_mix_filtrado_1, x_train_mix_filtrado_2, x_train_mix_orig, x_train, x_train_1, y_train, y_train_1,predictor)


        # Begin PRINT ==================================================================
            # Parameters -----------------------------------------------------------------
        num_row = 1 #2                                                                  # Number of rows per group
        num_col = 10 #8 #10                                                                 # Number of columns per group
        num_pixels = 28
        num_functions = 9                                                               # Number of functions to be displayed (=num_row_group*num_col_group)
        num_row_group = 9                                                               # Number of group rows
        num_col_group = 1                                                               # Number of group columns
        scale_factor = 1.0                                                              # Image scale factor
        figsize_x = num_col * num_col_group * scale_factor                              # Total width of a row
        figsize_y = num_row * num_row_group * scale_factor                              # Total height of a column
            # Images ---------------------------------------------------------------------
        #img_group = tf.stack([x_train_mix_orig, x_train, x_train_1, x_train_mix_filtrado_1, x_train_mix_filtrado_2, x_train_decoded_1, x_train_decoded_2, x__x, x_train_best_predicted_1])
        img_group = tf.stack([x_train_mix_orig, x_train, x_train_1, x_train_mix_filtrado_1, x_train_mix_filtrado_2, x_train_decoded_1, x_train_decoded_2, x__x])
        
            # Tags -----------------------------------------------------------------------
        #e_img = tf.stack(["x_mix_orig", "x_train", "x_train_1", "x_filt_1", "x_filt_2", "x_deco_1", "x_deco_2", "x__x", "x_best_pred"])
        e_img = tf.stack(["x_mix_orig", "x_train", "x_train_1", "x_filt_1", "x_filt_2", "x_deco_1", "x_deco_2", "x__x"])
        
            # Labels ---------------------------------------------------------------------
        print(y_train.shape)
        print(y_train_1.shape)
        
        labels_group = tf.stack([[y_train, y_train_1]])
        labels_index = [0]                                                              # rows with labels
            # Plot images ----------------------------------------------------------------
        photo_group(num_row, num_col, figsize_x, figsize_y, num_pixels,
                    num_functions, num_row_group, num_col_group,
                    img_group, e_img, labels_group, labels_index)
        plt.show()
        print("Fig.: En la primera fila se observan las imágenes de TRAIN superpuestas, las componentes en las dos siguientes,")
        print("      la reconstrucción final en la cuarta, la mejor imagen original basada en MSE en la quinta y en la última")
        print("      la mejor imagen según la predicción.")
        # End PRINT ====================================================================
        
        