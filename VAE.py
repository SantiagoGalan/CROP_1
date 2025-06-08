import numpy as np
import tensorflow as tf
from keras.layers import Lambda, Input, Dense, Concatenate
from keras.models import Model
from keras import backend as K
from ReparameterizationTrick import Sampling  # Clase para el truco de reparametrización
from keras.utils import plot_model
from IPython.display import Image, display

def Vae(encoder, decoder, show_model=False):
    """
    Construye el modelo VAE condicional combinando encoder y decoder.
    Permite que el decoder reciba una condición diferente (pero de igual dimensión) a la del encoder.
    Si al llamar al modelo se pasan solo dos condiciones, se usará la misma para encoder y decoder.

    Parámetros:
    -----------
    encoder : keras.Model
        Modelo encoder que recibe una imagen y una condición, y devuelve z_mean, z_log_var y z.
    decoder : keras.Model
        Modelo decoder que recibe un vector latente y una condición, y devuelve la imagen reconstruida.
    show_model : bool
        Si True, guarda y muestra el diagrama del modelo.

    Retorna:
    --------
    vae : keras.Model
        Modelo VAE completo que recibe [imagen, condición_encoder, condición_decoder] y devuelve la imagen reconstruida.
        Si se pasan solo dos entradas ([imagen, condición]), la condición se usará para ambos.
    """

    # Entradas del encoder
    encoder_inputs = encoder.input[0]  # Imagen de entrada
    encoder_cond = encoder.input[1]    # Condición para el encoder (one-hot)

    # Nueva entrada para la condición del decoder (misma dimensión que encoder_cond)
    decoder_cond = tf.keras.Input(shape=encoder_cond.shape[1:], name="decoder_condition")

    # Salidas del encoder (z_mean, z_log_var, z)
    z_mean, z_log_var, z = encoder([encoder_inputs, encoder_cond])

    # El decoder toma el vector latente z y la condición del decoder, y produce la imagen reconstruida
    vae_outputs = decoder([z, decoder_cond])

    # Instancia el modelo VAE completo
    vae = Model(inputs=[encoder_inputs, encoder_cond, decoder_cond], outputs=vae_outputs, name="vae")
    vae.summary()

    # Visualización opcional del modelo
    if show_model:
        from keras.utils import plot_model
        from IPython.display import Image, display
        plot_model(vae, to_file="vae.png", show_shapes=True, show_layer_names=True)
        display(Image(filename="vae.png"))

    # Wrapper para permitir pasar solo dos entradas (imagen y condición)
    def vae_predict(inputs, *args, **kwargs):
        # Si solo se pasan dos entradas, usa la misma condición para encoder y decoder
        if isinstance(inputs, list) and len(inputs) == 2:
            img, cond = inputs
            return vae.predict([img, cond, cond], *args, **kwargs)
        else:
            return vae.predict(inputs, *args, **kwargs)

    vae.predict_with_default_cond = vae_predict

    return vae