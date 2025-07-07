import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import BatchNormalization, Dropout, Dense, Layer

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


def build_predictor_model(image_size=28, n_cond=10):
    """
    Construye y devuelve el modelo predictor convolucional.

    Args:
        image_size (int): tamaño de la imagen (alto y ancho).
        n_cond (int): número de clases de salida (condiciones).

    Returns:
        keras.Model: modelo compilado predictor.
    """
    original_dim_C = (image_size, image_size, 1)
    original_dim = image_size * image_size

    input_predictor = Input(shape=(original_dim,), name="original_input")
    input_predictor_C = ReshapeLayer(original_dim_C)(input_predictor)

    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(input_predictor_C)
    x = BatchNormalization()(x)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = layers.Flatten()(x)
    x = Dropout(0.5)(x)
    predictor_outputs = Dense(n_cond, activation="softmax")(x)

    model = Model(inputs=input_predictor, outputs=predictor_outputs, name="predictor_C2")
    return model
