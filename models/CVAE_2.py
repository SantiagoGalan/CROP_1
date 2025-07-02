from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Lambda
import tensorflow as tf


def build_vae(encoder, decoder, img_shape=(28, 28, 1), cond_dim=10, beta=1.0):
    """
    VAE funcional que recibe:
    - imagen
    - condición_encoder
    - condición_decoder
    e incorpora pérdida KL con add_loss simbólica dentro del grafo.
    """
    # Entradas
    img_input = Input(shape=img_shape, name="image_input")
    cond_enc_input = Input(shape=(cond_dim,), name="encoder_condition")
    cond_dec_input = Input(shape=(cond_dim,), name="decoder_condition")

    # Encoder
    z_mean, z_log_var, z = encoder([img_input, cond_enc_input])

    # KL loss dentro de una capa Lambda que no modifica flujo, pero agrega pérdida
    def kl_with_loss(args):
        z_mean, z_log_var = args
        kl = -0.5 * tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1
        )
        
        kl = tf.reduce_mean(kl)
        tf.keras.backend.add_loss(beta * kl)
        return z_mean  # devuelve algo arbitrario

    _ = Lambda(kl_with_loss, output_shape=(None, z_mean.shape[-1]))([z_mean, z_log_var])


    # Decoder
    x_rec = decoder([z, cond_dec_input])

    # Modelo final
    vae_model = Model(inputs=[img_input, cond_enc_input, cond_dec_input],
                      outputs=x_rec,
                      name="vae")

    return vae_model
