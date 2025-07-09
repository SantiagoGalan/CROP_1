import numpy as np
import tensorflow as tf
from keras.datasets import mnist

def get_mnist_data(show_data=False, MIX="NONE"):

    '''
    if MIX == "MAX":
        maximum_image = np.maximum(x_train,x_train_1)
        x_train_mix = maximum_image                                                    # Habilta para MAX - Inabilita para AVERAGE
        print("x_train_1.shape:   ", x_train_1.shape)
        print("y_train_1.shape:     ", y_train_1.shape)
        print("x_train_mix.shape: ", x_train_mix.shape)
        maximum_image = np.maximum(x_test,x_test_1)
        x_test_mix = maximum_image                                                    # Habilta para MAX - Inabilita para AVERAGE
        print("x_test_1.shape:   ", x_test_1.shape)
        print("y_test_1.shape:     ", y_test_1.shape)
        print("x_test_mix.shape: ", x_test_mix.shape)

    ## Superimposed digits - AVERAGE

    if MIX == "AVERAGE":
        average_image = (x_train.astype(np.float32) + x_train_1.astype(np.float32)) / 2
        average_image = average_image.astype(np.uint8)  # Convert the pixel values back to uint8
        x_train_mix = average_image                                                      # Inhabilta para MAX - Habilita para AVERAGE
        print("x_train_1.shape:   ", x_train_1.shape)
        print("y_train_1.shape:     ", y_train_1.shape)
        print("x_train_mix.shape: ", x_train_mix.shape)
        average_image = (x_test.astype(np.float32) + x_test_1.astype(np.float32)) / 2
        average_image = average_image.astype(np.uint8)  # Convert the pixel values back to uint8
        x_test_mix = average_image                                                      # Inhabilta para MAX - Habilita para AVERAGE
        print("x_test_1.shape:   ", x_test_1.shape)
        print("y_test_1.shape:     ", y_test_1.shape)
        print("x_test_mix.shape: ", x_test_mix.shape)

    #x_train: Imagenes escaladas 1D.
    #y_train: Vector one-hot encoded.
    '''
    # MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    #print("x_train(60k).shape:      ", x_train.shape)

    # Normalization
    #image_size = x_train.shape[1]                                                   # 28
    x_train = x_train.astype('float32') / 255                                       # [0, 1] imagnes re-escalada
    x_test = x_test.astype('float32') / 255                                         # [0, 1]
    
    #x_train = np.reshape(x_train, [-1, original_dim])                               # imagenes en 1D
    #x_test = np.reshape(x_test, [-1, original_dim])

    y_train = tf.keras.utils.to_categorical(y_train)                                # one-hot
    y_test = tf.keras.utils.to_categorical(y_test)                                  #
    #y_test_orig = y_test

    # Split training data into training and validation sets
    x_train, x_val = x_train[:55000], x_train[55000:]
    y_train, y_val = y_train[:55000], y_train[55000:]


    # Original individual images for Convolutional NN (28x28)
   # original_dim_C = image_size                                                     # 28
   # x_train_C = np.expand_dims(x_train, -1)
   # x_test_C = np.expand_dims(x_test, -1)
    # Split training data into training and validation sets
    #x_train_C, x_val_C = x_train_C[:55000], x_train_C[55000:]
    #print("x_train_C.shape:    ", x_train_C.shape)
    #print("x_val_C.shape:    ", x_val_C.shape)
    #print("x_test_C.shape:     ", x_test_C.shape)

    # Flatten the individual images for Dense NN (784)
    #original_dim = image_size * image_size                                          # 784 cantidad de pixeles de la imagen
    
    #print("x_train.shape:      ", x_train.shape)
    #print("x_val.shape:      ", x_val.shape)
    #print("x_test.shape:      ", x_test.shape)

    # Ver para condicionar Convolutional  ******************************************


    ## Superimposed digits - MAX
    np.random.seed(3333)                                    #3333 Cambio el seed de 2022 (2024) para ver variabilidad   # de VAE 5 para fijar las pruebas y poder comparar
    permrows = np.random.permutation(x_train.shape[0])
    #x_train_C_1 = x_train_C[permrows,:]                                               # alternative set for Convolutional
    x_train_1 = x_train[permrows,:]                                               # alternative set for Dense
    y_train_1 = y_train[permrows,:]
    #permrows = np.random.permutation(x_test.shape[0])
    #x_test_C_1 = x_test_C[permrows,:]                                               # alternative set for Convolutional
    #x_test_1 = x_test[permrows,:]                                               # alternative set for Dense
    #y_test_1 = y_test[permrows,:]


    return x_train,x_test, x_val, y_train, y_test, y_val,x_train_1,y_train_1