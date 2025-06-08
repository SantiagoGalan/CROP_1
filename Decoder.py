import numpy as np
import tensorflow as tf
from keras.layers import Lambda, Input, Dense, Concatenate, Reshape
from keras.models import Model
from keras import backend as K
from ReparameterizationTrick import Sampling  # Clase para el truco de reparametrización
from keras.utils import plot_model
from IPython.display import Image, display

def decoder(x_train, y_train, intermediate_dim=128, latent_dim=2, show_model=False):
    """
    Construye el modelo decoder para un VAE condicional.

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
    decoder : keras.Model
        Modelo decoder que recibe un vector latente y una condición, y devuelve la imagen reconstruida.
    """

    # Obtiene la cantidad de condiciones y la forma de la imagen
    n_cond = y_train.shape[1]
    img_shape = x_train.shape[1:]  # Por ejemplo, (28, 28)
    output_units = np.prod(img_shape)

    # Definición de entradas
    latent_inputs = Input(shape=(latent_dim,), name="z_sampling")  # Vector latente z
    cond_decoder = Input(shape=(n_cond,), name="decoder_condition")  # Condición (one-hot)
    # Concatenación de z y condición
    expanded_inputs = Concatenate()([latent_inputs, cond_decoder])

    # Capa oculta intermedia
    x = Dense(intermediate_dim, activation="relu")(expanded_inputs)
    # Capa de salida: tantas unidades como píxeles de la imagen, activación sigmoide para valores entre 0 y 1
    x = Dense(output_units, activation="sigmoid")(x)
    # Remodela la salida a la forma original de la imagen + canal
    output = Reshape((28, 28, 1))(x)
    #output = Reshape(img_shape + (1,))(x)

    # Instancia el modelo decoder
    decoder = Model(inputs=[latent_inputs, cond_decoder], outputs=output, name="decoder")
    decoder.summary()

    # Visualización opcional del modelo
    if show_model:
        from keras.utils import plot_model
        from IPython.display import Image, display
        plot_model(decoder, to_file="decoder.png", show_shapes=True, show_layer_names=True)
        display(Image(filename="decoder.png"))

    return decoder

