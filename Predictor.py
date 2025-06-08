from tensorflow.keras.layers import Layer
import numpy as np
import tensorflow as tf
from keras.layers import Lambda, Input, Dense, Concatenate, Conv2D, Flatten
from keras.models import Model
from keras import backend as K
from ReparameterizationTrick import Sampling  # Asegúrate de que el archivo esté en el mismo directorio
# Si usas MNIST o matplotlib, descomenta las siguientes líneas:
# from keras.datasets import mnist
# import matplotlib.pyplot as plt
from keras.utils import plot_model
from IPython.display import Image, display

import ReshapeLayer
    ## Predictor CONVOLUCIONAL ---------------------------------------------------------------------

def predictor():
    # network parameters
    original_dim_C = (28, 28, 1)                                                       # ¿agregar condición?
    n_cond = 10 # parametrizar con x_Train y y_train 
    # Define the input layer
    # Define predictor model ---------------------------------------------------------
    input_predictor = Input(shape=original_dim_C, name="original_input")

    # Use the custom reshape layer
    input_predictor_C = ReshapeLayer.ReshapeLayer(original_dim_C)(input_predictor)

    predictor_inputs = input_predictor_C

    x = Conv2D(32, 3, activation="relu", strides=2, padding="same")(predictor_inputs) #por que 32 y 64?
    x = Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = Flatten()(x)
    predictor_outputs = Dense(n_cond, activation="softmax")(x)


    # instantiate decoder model

    predictor_C = Model(inputs=input_predictor, outputs=predictor_outputs, name="predictor")
    predictor_C.summary()
    
    plot_model(predictor_C,to_file="predictor.png", show_shapes=True, show_layer_names=True)

    display(Image(filename="predictor.png"))
    return predictor_C