import numpy as np
import tensorflow as tf
from keras.layers import Lambda, Input, Dense, Concatenate
from keras.models import Model
from keras import backend as K
from ReparameterizationTrick import Sampling  # Asegúrate de que el archivo esté en el mismo directorio
# Si usas MNIST o matplotlib, descomenta las siguientes líneas:
# from keras.datasets import mnist
# import matplotlib.pyplot as plt
from keras.utils import plot_model
from IPython.display import Image, display

## VAE Model -------------------------------------------------------------------

  # Define VAE model -------------------------------------------------------
'''
def Vae(encoder,decoder):
    
    vae_outputs = decoder([encoder(encoder.inputs),decoder.get_layer("decoder_condition")])

    # instantiate VAE model
    vae = Model(inputs=[encoder.inputs, decoder.get_layer("decoder_condition")], outputs=vae_outputs, name="vae")
    vae.summary()
    return vae
'''
def Vae(encoder, decoder):
    # Inputs
    encoder_inputs = encoder.input[0]  # inputs_images
    encoder_cond = encoder.input[1]    # encoder_condition

    # Salidas del encoder
    z_mean, z_log_var, z = encoder([encoder_inputs, encoder_cond])

    # Inputs del decoder (deben ser los mismos que los del encoder)
    decoder_cond = encoder_cond  # Usamos la misma condición

    # Salida del decoder
    vae_outputs = decoder([z, decoder_cond])

    # Modelo VAE
    vae = Model(inputs=[encoder_inputs, encoder_cond], outputs=vae_outputs, name="vae")
    vae.summary()
    plot_model(vae, show_shapes=True, show_layer_names=True)
    
    plot_model(vae,to_file="vae.png", show_shapes=True, show_layer_names=True)

    display(Image(filename="vae.png"))    
    
    
    return vae