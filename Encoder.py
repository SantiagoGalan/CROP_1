import numpy as np
import tensorflow as tf
from keras.layers import Lambda, Input, Dense, Concatenate, Flatten
from keras.models import Model
from keras import backend as K
from ReparameterizationTrick import Sampling  # Clase para el truco de reparametrización
from keras.utils import plot_model
from IPython.display import Image, display

def encoder(x_train, y_train, intermediate_dim=128, latent_dim=2, show_model=False): 

    #intermediate_dim= latent_dim*2
    """
    Construye el modelo encoder para un VAE condicional.

    Parámetros:
    -----------
    x_train : np.ndarray
        Imágenes de entrenamiento (esperado shape: [N, alto, ancho])
    y_train : np.ndarray
        Etiquetas codificadas one-hot (shape: [N, n_cond])
    intermediate_dim : int
        Número de neuronas en la capa oculta intermedia.
    latent_dim : int
        Dimensión del espacio latente.
    show_model : bool
        Si True, guarda y muestra el diagrama del modelo.

    Retorna:
    --------
    encoder : keras.Model
        Modelo encoder que recibe una imagen y una condición, y devuelve z_mean, z_log_var y z (muestreo).
    """

    # Parámetros de entrada
    img_shape = x_train.shape[1:]
    n_cond = y_train.shape[1]

    # Definición de entradas
    # input_img_flat = Input(shape=(np.prod(img_shape),), name="input_img_flat")
    input_img = Input(shape=img_shape, name="input_img")
    input_img_flat = Flatten()(input_img)
    cond_encoder = Input(shape=(n_cond,), name="encoder_condition")

    # Concatenación de imagen y condición
    expanded_inputs = Concatenate()([input_img_flat, cond_encoder])

    # Capa oculta intermedia
    x = Dense(intermediate_dim, activation="relu")(expanded_inputs)

    # Salidas del encoder: media y log-varianza del espacio latente
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)

    # Muestreo usando el truco de reparametrización
    z = Sampling()((z_mean, z_log_var))

    # Instancia el modelo encoder
    encoder = Model(inputs=[input_img, cond_encoder], outputs=[z_mean, z_log_var, z], name="encoder")
    encoder.summary()

    # Visualización opcional
    if show_model:
        from keras.utils import plot_model
        from IPython.display import Image, display
        plot_model(encoder, to_file="encoder.png", show_shapes=True, show_layer_names=True)
        display(Image(filename="encoder.png"))

    return encoder