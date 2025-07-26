
from keras.layers import Layer
import tensorflow as tf
from keras.saving import register_keras_serializable

@register_keras_serializable()
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

