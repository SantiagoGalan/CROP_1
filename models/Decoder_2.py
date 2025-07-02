import tensorflow as tf
from tensorflow.keras import layers, Model
from layers.ReshapeLayer import ReshapeLayer

def build_decoder(
    img_shape=(28,28,1),
    cond_dim=10,
    intermediate_dim=128,
    latent_dim=2,
    show_model=False
):
    """
    Construye el decoder del VAE condicional.

    Args:
        img_shape: tupla con la forma de la imagen (H, W, C).
        cond_dim: dimensión de la condición one-hot.
        intermediate_dim: neuronas en la capa densa intermedia.
        latent_dim: dimensión del espacio latente.
        show_model: si True, muestra el diagrama del modelo.

    Returns:
        decoder: keras.Model que recibe [z, condición] y devuelve la imagen reconstruida
    """
    # Entradas
    z_input = layers.Input(shape=(latent_dim,), name="decoder_z")
    cond_input = layers.Input(shape=(cond_dim,), name="decoder_condition")

    # Concatenar latente con condición
    x = layers.Concatenate(name="concat_z_cond")([z_input, cond_input])
    x = layers.Dense(intermediate_dim, activation="relu", name="decoder_dense")(x)

    # Salida a vector plano de imagen
    flat_units = img_shape[0] * img_shape[1] * img_shape[2]
    x = layers.Dense(flat_units, activation="sigmoid", name="decoder_output_flat")(x)

    # Reshape a forma de imagen
    x = ReshapeLayer(img_shape)(x)

    decoder = Model(
        inputs=[z_input, cond_input],
        outputs=x,
        name="decoder"
    )

    if show_model:
        tf.keras.utils.plot_model(decoder, to_file="decoder.png", show_shapes=True)
        from IPython.display import Image, display
        display(Image(filename="decoder.png"))

    return decoder
