from tensorflow.keras.layers import Layer
import tensorflow as tf

class ReshapeLayer(Layer):
    """
    Capa personalizada para aplicar un reshape a la entrada en modelos Keras.

    Parámetros:
    -----------
    target_shape : tuple
        Nueva forma deseada para la entrada (sin incluir el batch).
    """
    def __init__(self, target_shape, **kwargs):
        super(ReshapeLayer, self).__init__(**kwargs)
        self.target_shape = target_shape

    def call(self, inputs):
        """
        Aplica reshape a la entrada.
        """
        return tf.reshape(inputs, [-1, *self.target_shape])

    def get_config(self):
        """
        Permite serializar la capa personalizada.
        """
        config = super().get_config().copy()
        config.update({
            'target_shape': self.target_shape,
        })
        return config