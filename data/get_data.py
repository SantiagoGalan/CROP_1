import numpy as np
import tensorflow as tf
from keras.datasets import mnist, fashion_mnist


def get_mnist_data(dataset="mnist"):

    # print("x_train(60k).shape:      ", x_train.shape)

    if dataset == "fashion":

        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        print(f"Usando {dataset} como dataset")

    else:
        # MNIST dataset
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        print(f"Usando {dataset} como dataset")
    # Normalization
    # image_size = x_train.shape[1]                                                   # 28
    x_train = x_train.astype("float32") / 255  # [0, 1] imagnes re-escalada
    x_test = x_test.astype("float32") / 255
    # [0, 1]
    image_size = x_train.shape[1]
    original_dim = image_size * image_size
    x_train = np.reshape(x_train, [-1, original_dim])  # imagenes en 1D
    x_test = np.reshape(x_test, [-1, original_dim])

    y_train = tf.keras.utils.to_categorical(y_train)  # one-hot
    y_test = tf.keras.utils.to_categorical(y_test)  #
    # y_test_orig = y_test

    # Split training data into training and validation sets
    x_train, x_val = x_train[:55000], x_train[55000:]
    y_train, y_val = y_train[:55000], y_train[55000:]

    np.random.seed(
        3333
    )  # 3333 Cambio el seed de 2022 (2024) para ver variabilidad   # de VAE 5 para fijar las pruebas y poder comparar
    permrows = np.random.permutation(x_train.shape[0])
    x_train_1 = x_train[permrows, :]  # alternative set for Dense
    y_train_1 = y_train[permrows, :]
    permrows = np.random.permutation(x_test.shape[0])
    x_test_1 = x_test[permrows, :]  # alternative set for Dense
    y_test_1 = y_test[permrows, :]

    return {
        "x_train": x_train,
        "x_test": x_test,
        "x_val": x_val,
        "y_train": y_train,
        "y_test": y_test,
        "y_val": y_val,
        "x_train_1": x_train_1,
        "y_train_1": y_train_1,
        "x_test_1": x_test_1,
        "y_test_1": y_test_1,
    }
