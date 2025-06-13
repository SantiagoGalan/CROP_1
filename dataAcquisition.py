import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from tensorflow.keras.datasets import fashion_mnist

def data_acquisition(MIX="AVERAGE"):
    # Carga el dataset MNIST (imágenes de dígitos manuscritos)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print("x_train(60k).shape:      ", x_train.shape)
    #

    # Carga el dataset Fashion MNIST (imágenes de ropa)
    #(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    #print("x_train(60k).shape:      ", x_train.shape)
    #print("y_train.shape:           ", y_train.shape)
    #print("x_test(10k).shape:       ", x_test.shape)



    # Normaliza las imágenes a valores entre 0 y 1
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Añade un canal para compatibilidad con redes convolucionales
    x_train_C = np.expand_dims(x_train, -1)
    x_test_C = np.expand_dims(x_test, -1)

    # Separa un conjunto de validación (últimos 5,000 ejemplos)
    x_train_C, x_val_C = x_train_C[:55000], x_train_C[55000:]
    print("x_train_C.shape:    ", x_train_C.shape)
    print("x_val_C.shape:    ", x_val_C.shape)
    print("x_test_C.shape:     ", x_test_C.shape)

    # Convierte las etiquetas a codificación one-hot
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    # Divide imágenes y etiquetas para entrenamiento y validación
    x_train, x_val = x_train[:55000], x_train[55000:]
    y_train, y_val = y_train[:55000], y_train[55000:]

    print("x_train.shape:      ", x_train.shape)
    print("x_val.shape:      ", x_val.shape)
    print("x_test.shape:      ", x_test.shape)

    # Crea conjuntos alternativos mezclando filas aleatoriamente (para experimentos de mezcla)
    np.random.seed(3333)
    permrows = np.random.permutation(x_train.shape[0])
    x_train_1 = x_train[permrows, :]
    y_train_1 = y_train[permrows, :]

    permrows = np.random.permutation(x_test.shape[0])
    x_test_1 = x_test[permrows, :]
    y_test_1 = y_test[permrows, :]

    # Mezcla imágenes de dos formas: máximo o promedio pixel a pixel
    if MIX == "MAX":
        x_train_mix = np.maximum(x_train, x_train_1)
        x_test_mix = np.maximum(x_test, x_test_1)
        print("x_train_mix.shape: ", x_train_mix.shape)
        print("x_test_mix.shape: ", x_test_mix.shape)
        average_image = None  # No se usa en este modo

    elif MIX == "AVERAGE":
        # Mezcla por promedio pixel a pixel
        average_image = ((x_train.astype(np.float32) + x_train_1.astype(np.float32)) / 2).astype(np.uint8)
        x_train_mix = average_image
        average_image = ((x_test.astype(np.float32) + x_test_1.astype(np.float32)) / 2).astype(np.uint8)
        x_test_mix = average_image
        print("x_train_mix.shape: ", x_train_mix.shape)
        print("x_test_mix.shape: ", x_test_mix.shape)
    else:
        raise ValueError("MIX debe ser 'MAX' o 'AVERAGE'")

    # Devuelve los conjuntos procesados para entrenamiento, validación y mezcla
    return x_train, x_val, y_train, y_val, average_image, x_train_mix, x_test_mix, x_train_1, y_train_1