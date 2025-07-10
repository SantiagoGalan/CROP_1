import numpy as np
import matplotlib.pyplot as plt

def condiciones(cvae, x_input):
    """
    Muestra cómo se reconstruye una imagen de entrada bajo las 10 condiciones posibles (0 a 9).
    """

    # Asegurar que x_input tenga batch dimension
    #if x_input.shape == (28, 28):
    x_input = np.expand_dims(x_input, axis=0)

    # Repetir la imagen 10 veces
    x_repeated = np.repeat(x_input, repeats=10, axis=0) 

    # Crear las 10 condiciones one-hot
    condiciones = np.eye(10)  # (10, 10)

    # Codificar
    z_mean, z_log_var, z = cvae.encoder.predict([x_repeated, condiciones])

    # Reconstruir
    reconstrucciones = cvae.decoder.predict([z, condiciones])

    # Mostrar
    plt.figure(figsize=(15, 2))
    for i in range(10):
        plt.subplot(1, 10, i + 1)
        plt.imshow(reconstrucciones[i].reshape(28, 28), cmap="gray")
        plt.title(f"Clase {i}")
        plt.axis("off")
    plt.suptitle("Reconstrucciones bajo distintas condiciones")
    plt.show()


def variantes(cvae, condicion_id, num_variantes=10):
    """
    Muestra múltiples imágenes generadas para una misma condición.
    
    Args:
        cvae: modelo CVAE entrenado.
        condicion_id: entero de 0 a 9, la clase condicional deseada.
        num_variantes: número de muestras a generar.
    """
    # Comprobar que el ID sea válido
    assert 0 <= condicion_id <= 9, "La condición debe estar entre 0 y 9."

    # Crear condición one-hot repetida
    condicion = np.eye(10)[condicion_id]
    condiciones = np.repeat([condicion], num_variantes, axis=0)  # (num_variantes, 10)

    # Generar z aleatorios ~ N(0,1)
    latent_dim = cvae.decoder.input_shape[0][1]  # obtiene la dimensión latente del input
    z = np.random.normal(size=(num_variantes, latent_dim))  # (num_variantes, latent_dim)

    # Generar imágenes con el decoder
    imgs_generadas = cvae.decoder.predict([z, condiciones])

    # Mostrar
    plt.figure(figsize=(15, 2))
    for i in range(num_variantes):
        plt.subplot(1, num_variantes, i + 1)
        plt.imshow(imgs_generadas[i].reshape(28, 28), cmap="gray")
        #plt.imshow(imgs_generadas[i], cmap="gray")
        plt.axis("off")
    plt.suptitle(f"Variantes generadas para la clase {condicion_id}")
    plt.show()
