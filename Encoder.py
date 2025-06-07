import numpy as np
import tensorflow as tf
from keras.layers import Lambda, Input, Dense, Concatenate, Flatten
from keras.models import Model
from keras import backend as K
from ReparameterizationTrick import Sampling  # Asegúrate de que el archivo esté en el mismo directorio
# Si usas MNIST o matplotlib, descomenta las siguientes líneas:
# from keras.datasets import mnist
# import matplotlib.pyplot as plt
from keras.utils import plot_model
from IPython.display import Image, display



def encoder(x_train, y_train ,intermediate_dim=128, latent_dim = 2):
    
    ## Encoder ---------------------------------------------------------------------
    # network parameters
    original_dim = np.shape(x_train)[1]
    n_cond = np.shape(y_train)[1]                                                   # n_cond = 10 (number of conditions)

    # Define encoder model -------------------------------------------------------
    input_img = Input(shape=(28, 28), name="input_img")
    input_img_flat = Flatten()(input_img)
    cond_encoder = Input(shape=(n_cond,), name="encoder_condition")
    expanded_inputs = Concatenate()([input_img_flat,cond_encoder])                   # se amplía la entrada "x" (784) con la condición "c" (10)

    x = Dense(intermediate_dim, activation="relu")(expanded_inputs)
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)

    # use reparameterization trick to push the sampling out as input
    z = Sampling()((z_mean, z_log_var))                                             # (z_mean, z_log_var) is the tuple

    # instantiate encoder model
    encoder = Model(inputs=[input_img,cond_encoder], outputs=[z_mean, z_log_var, z], name="encoder")
    encoder.summary()

    plot_model(encoder,to_file="encoder.png", show_shapes=True, show_layer_names=True)

    display(Image(filename="encoder.png"))
    return encoder
