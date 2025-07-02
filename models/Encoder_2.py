import tensorflow as tf
from tensorflow.keras import layers, Model


def build_encoder(img_shape=(28,28,1), cond_dim=10, intermediate_dim=128, latent_dim=2, show_model=False):
    """
    Construye el encoder del VAE condicional.

    Args:
        img_shape: tupla con la forma de la imagen (H, W, C).
        cond_dim: dimensión de la condición one-hot.
        intermediate_dim: neuronas en la capa densa intermedia.
        latent_dim: dimensión del espacio latente.
        show_model: si True, muestra el diagrama del modelo.

    Returns:
        encoder: keras.Model que recibe [imagen, condición] y devuelve [z_mean, z_log_var, z]
    """
    # Entradas
    img_input = layers.Input(shape=img_shape, name="encoder_image")
    cond_input = layers.Input(shape=(cond_dim,), name="encoder_condition")

    # Aplanado y concatenación
    x = layers.Flatten(name="flattened_image")(img_input)
    x = layers.Concatenate(name="concat_image_cond")([x, cond_input])

    # Capa oculta
    x = layers.Dense(intermediate_dim, activation="relu", name="encoder_dense")(x)

    # Parámetros latentes
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

    # Muestreo estocástico usando Lambda para evitar operaciones tf.* directas
    def sample_z(args):
        z_m, z_lv = args
        batch = tf.shape(z_m)[0]
        epsilon = tf.random.normal(shape=(batch, latent_dim))
        return z_m + tf.exp(0.5 * z_lv) * epsilon

    z = layers.Lambda(sample_z, name="z")([z_mean, z_log_var])

    encoder = Model(
        inputs=[img_input, cond_input],
        outputs=[z_mean, z_log_var, z],
        name="encoder"
    )

    if show_model:
        tf.keras.utils.plot_model(encoder, to_file="encoder.png", show_shapes=True)
        from IPython.display import Image, display
        display(Image(filename="encoder.png"))

    return encoder
