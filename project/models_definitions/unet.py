import tensorflow as tf
from tensorflow.keras import layers, Model



def expand_cond(cond, filters):
    c = layers.Dense(filters, activation="relu")(cond)
    c = layers.Reshape((1, 1, filters))(c)
    return c

def conv_block(x, cond, filters):
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = FiLM(filters)(x, cond)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = FiLM(filters)(x, cond)
    x = layers.Activation("relu")(x)

    return x

def build_film_unet_mnist(cond_dim=10, base_filters=32):
    img_in  = tf.keras.Input(shape=(28, 28, 1))
    cond_in = tf.keras.Input(shape=(cond_dim,))

    # Encoder
    c1 = conv_block(img_in, cond_in, base_filters)
    p1 = layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, cond_in, base_filters * 2)
    p2 = layers.MaxPooling2D()(c2)

    # Bottleneck
    b = conv_block(p2, cond_in, base_filters * 4)

    # Decoder
    u2 = layers.UpSampling2D()(b)
    u2 = layers.Concatenate()([u2, c2])
    c3 = conv_block(u2, cond_in, base_filters * 2)

    u1 = layers.UpSampling2D()(c3)
    u1 = layers.Concatenate()([u1, c1])
    c4 = conv_block(u1, cond_in, base_filters)

    out = layers.Conv2D(1, 1, activation="sigmoid")(c4)

    return tf.keras.Model([img_in, cond_in], out)

class FiLM(layers.Layer):
    def __init__(self, channels):
        super().__init__()
        self.dense = layers.Dense(channels * 2)

    def call(self, x, cond):
        gamma_beta = self.dense(cond)
        gamma, beta = tf.split(gamma_beta, 2, axis=-1)

        gamma = tf.reshape(gamma, (-1, 1, 1, x.shape[-1]))
        beta  = tf.reshape(beta,  (-1, 1, 1, x.shape[-1]))

        return gamma * x + beta


    def call(self, x, cond):
        gamma_beta = self.dense(cond)
        gamma, beta = tf.split(gamma_beta, 2, axis=-1)
        gamma = tf.reshape(gamma, (-1, 1, 1, x.shape[-1]))
        beta  = tf.reshape(beta,  (-1, 1, 1, x.shape[-1]))
        return gamma * x + beta