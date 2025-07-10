import numpy as np
import tensorflow as tf
import inferencias.best_digit  as bd
import inferencias.outcomes as out
from inferencias.fotos import photo_group
import matplotlib.pyplot as plt

import importlib
importlib.reload(bd)
import importlib
importlib.reload(out)


def separar_digitos(x_train,x_train_1,y_train,y_train_1,encoder,decoder,predictor):
        
    #-------------------------------versión 3 (TRAIN a simplificar CORTO PARA PRUEBAS CON VARIANZA)-----------------------------------------
    #                     REVISAR este help
    # 11 minutos
    # ------------------------------------------------------------------------------
    # Encode and decode some digits

    ## Superimposed digits - MAX
    #maximum_image = np.maximum(x_train,x_train_1)
    # x_train_mix = maximum_image                                                    # Habiltar para MAX - Inabilitar para AVERAGE

    ## Superimposed digits - AVERAGE
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


    Iterations = 5

    bias = 0.22
    slope = 22.

    beta = 1.
    alpha_1 = -2
    alpha_2 = -22

    print("Shape de x_train")
    print(x_train.shape)

    for j in range(Iterations):

        # best_digit_var_sigmoid(x_mix_filtrado_2, x_mix_orig, alpha, bias, slope)

        x_train_mix_filtrado_1, x_train_decoded_1 = bd.best_digit_var_sigmoid(x_train_mix_filtrado_2, x_train_mix_orig, alpha_2, bias, slope,encoder,decoder,predictor)
        alpha_2 = alpha_2 * beta

        x_train_mix_filtrado_2, x_train_decoded_2 = bd.best_digit_var_sigmoid(x_train_mix_filtrado_1, x_train_mix_orig, alpha_1, bias, slope,encoder,decoder,predictor)
        alpha_1 = alpha_1 * beta

        ######################################################


        print("ITERACIÓN A: ", j)

        x_train_best_predicted_1, y_train_predicted_1_f, y_train_predicted_2_f = out.outcomes(x_train_decoded_1, x_train_decoded_2, x_train_mix_filtrado_1,
                                                                                            x_train_mix_filtrado_2,x_train_mix_orig, x_train, x_train_1,
                                                                                            y_train, y_train_1,predictor)


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
        
        print(x_train_mix_orig.shape)
        
        
        
        img_group = tf.stack([x_train_mix_orig,x_train,x_train_1,x_train_mix_filtrado_1,x_train_mix_filtrado_2,
                              x_train_decoded_1,x_train_decoded_2,x__x,x_train_best_predicted_1])
        
        
        #img_group = tf.stack([tf.reshape(x_train_mix_orig,(x_train_mix_orig.shape[0],x_train_mix_orig.shape[1]*x_train_mix_orig.shape[2])), 
        #                      tf.reshape(x_train,(x_train.shape[0],x_train.shape[1]*x_train.shape[2])), 
        #                      tf.reshape(x_train_1,(x_train_1.shape[0],x_train_1.shape[1]*x_train_1.shape[2])), 
        #                      tf.reshape(x_train_mix_filtrado_1,(x_train_mix_filtrado_1.shape[0],x_train_mix_filtrado_1.shape[1]*x_train_mix_filtrado_1.shape[2])), 
        #                      tf.reshape(x_train_mix_filtrado_2,(x_train_mix_filtrado_2.shape[0],x_train_mix_filtrado_2.shape[1]*x_train_mix_filtrado_2.shape[2])), 
        #                      tf.reshape(x_train_decoded_1,(x_train_decoded_1.shape[0],x_train_decoded_1.shape[1]*x_train_decoded_1.shape[2])), 
        #                      tf.reshape(x_train_decoded_2,(x_train_mix_orig.shape[0],x_train_mix_orig.shape[1]*x_train_mix_orig.shape[2])), 
        #                      tf.reshape(x__x,(x__x.shape[0],x__x.shape[1]*x__x.shape[2])), 
        #                      x_train_best_predicted_1
        #])
            # Tags -----------------------------------------------------------------------
        e_img = tf.stack(["x_mix_orig", "x_train", "x_train_1", "x_filt_1", "x_filt_2", "x_deco_1", "x_deco_2", "x__x", "x_best_pred"])
            # Labels ---------------------------------------------------------------------
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


