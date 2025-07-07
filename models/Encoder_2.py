import tensorflow as tf
from tensorflow.keras import layers, Model
from layers import Input, Concatenate, Dense, Lamda

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
   
    
    # Define encoder model -------------------------------------------------------

    original_inputs = Concatenate()([input_encoder,cond_encoder])                   # se amplía la entrada "x" (784) con la condición "c" (10)

    x = Dense(intermediate_dim, activation="relu")(original_inputs)
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)

    # use reparameterization trick to push the sampling out as input
    z = Sampling()((z_mean, z_log_var))                                             # (z_mean, z_log_var) is the tuple

    encoder = Model(inputs=original_inputs, outputs=[z_mean, z_log_var, z], name="encoder")
   """   
    
    ########################
    # Entradas
    img_input = Input(shape=(img_shape,), name="encoder_image")
    cond_input = Input(shape=(cond_dim,), name="encoder_condition")


    # Aplanado y concatenación
    #x = layers.Flatten(name="flattened_image")(img_input)
    inputs_ampliados = Concatenate(name="concat_image_cond")([img_input, cond_input])

    # Capa oculta
    x = Dense(intermediate_dim, activation="relu", name="encoder_dense")(inputs_ampliados)

    # Parámetros latentes
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)

    class Sampling(layers.Layer):
        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch = tf.shape(z_mean)[0]                                                 # batch = number of data in the batch
            dim = tf.shape(z_mean)[1]                                                   # dim   = number of dimensions of "z"
            # by default, random_normal has mean=0 and std=1.0
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.keras.backend.exp(0.5 * z_log_var) * epsilon
    
    z  = Sampling()((z_mean,z_log_var))
    
    encoder = Model(
        inputs=inputs_ampliados,
        outputs=[z_mean, z_log_var, z],
        name="encoder"
    )

    if show_model:
        tf.keras.utils.plot_model(encoder, to_file="encoder.png", show_shapes=True)
        from IPython.display import Image, display
        display(Image(filename="encoder.png"))

    return encoder
