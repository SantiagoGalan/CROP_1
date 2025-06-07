import numpy as np
import tensorflow as tf
from keras.layers import Lambda, Input, Dense, Concatenate, Reshape
from keras.models import Model
from keras import backend as K
from ReparameterizationTrick import Sampling  # Asegúrate de que el archivo esté en el mismo directorio
# Si usas MNIST o matplotlib, descomenta las siguientes líneas:
# from keras.datasets import mnist
# import matplotlib.pyplot as plt
from keras.utils import plot_model
from IPython.display import Image, display


def decoder(x_train, y_train, intermediate_dim=128, latent_dim=2): #Transformar a una clase con atributos inputs (desglosado), outputs y modelo. gets
        
    n_cond = np.shape(y_train)[1]                                                   # n_cond = 10 (number of conditions) ¿se repite del encoder?
    #original_dim = np.shape(x_train)[1]
    # Define decoder model -------------------------------------------------------
    latent_inputs = Input(shape=(latent_dim,), name="z_sampling")
    cond_decoder = Input(shape=(n_cond,), name="decoder_condition")
    expanded_inputs = Concatenate()([latent_inputs,cond_decoder])

    x = Dense(intermediate_dim, activation="relu")(expanded_inputs)
    x = Dense(784, activation="sigmoid")(x)
    output = Reshape((28, 28, 1))(x)
    
    # instantiate decoder model
    
    decoder = Model(inputs=[latent_inputs,cond_decoder], outputs=output, name="decoder")
    #decoder = Model(inputs=latent_inputs, outputs=decoder_outputs, name="decoder")
    decoder.summary()

    #plot_model(decoder, to_file='vae_D_decoder.png', show_shapes=True)
    plot_model(decoder, show_shapes=True, show_layer_names=True)

    plot_model(decoder,to_file="decoder.png", show_shapes=True, show_layer_names=True)

    display(Image(filename="decoder.png"))    

    return decoder

