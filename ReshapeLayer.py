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

class ReshapeLayer(Layer):
    def __init__(self, target_shape, **kwargs):
        super(ReshapeLayer, self).__init__(**kwargs)
        self.target_shape = target_shape

    def call(self, inputs):
        return tf.reshape(inputs, [-1, *self.target_shape])

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'target_shape': self.target_shape,
        })
        return config
    