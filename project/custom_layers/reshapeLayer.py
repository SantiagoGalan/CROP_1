
import keras
from keras.layers import Layer
import tensorflow as tf
@keras.saving.register_keras_serializable()
class ReshapeLayer(Layer):
    def __init__(self, target_shape, **kwargs):
        super(ReshapeLayer, self).__init__(**kwargs)
        self.target_shape = target_shape

    def call(self, inputs):
        return tf.reshape(inputs, [-1, *self.target_shape])