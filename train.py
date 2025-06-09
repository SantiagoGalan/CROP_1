import numpy as np
import tensorflow as tf
from keras.layers import  Concatenate
from keras.datasets import mnist
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
mse = MeanSquaredError()
binary_crossentropy = BinaryCrossentropy()
from keras import backend as K
import matplotlib as mplt
import matplotlib.pyplot as plt
from keras.utils import to_categorical


# Actually training the model.
def train(model, x_train, y_train, x_val, y_val):
    #cp_callback = cp_callback
    #cp_callback = cp_callback_32
    #cp_callback = cp_callback_128
    #cp_callback = cp_callback_256                              #parametrizar para automatizar

    # Data
    input_encoder = x_train
    cond_encoder = y_train
    cond_decoder = y_train  # Puedes cambiar por otra condición si quieres experimentar
    estimated_output = x_train

    input_encoder_val = x_val
    cond_encoder_val = y_val
    cond_decoder_val = y_val  # Puedes cambiar por otra condición si quieres experimentar
    estimated_output_val = x_val

    # Concatenate input and condition for both training and validation data
    #original_inputs = np.concatenate([x_train, y_train], axis=1)
    #original_inputs_val = np.concatenate([x_val, y_val], axis=1)

    #original_inputs = Concatenate()([input_encoder, cond_encoder])
    #original_inputs_val = Concatenate()([input_encoder_val, cond_encoder_val])

    #print("Shapes of training inputs:")
    #print(f"original_inputs: {original_inputs.shape}")
    #print(f"y_train: {y_train.shape}")
    #print("Shapes of validation inputs:")
    #print(f"original_inputs_val: {original_inputs_val.shape}")
    #print(f"y_val: {y_val.shape}")

    # Fit the model with early stopping and checkpoint callbacks
        # Training
    model.fit(
        x=[x_train, y_train, y_train],  # [imagen, condición_encoder, condición_decoder]
        y=estimated_output,
        batch_size=128,
        epochs=1,
        validation_data=([x_val, y_val, y_val], estimated_output_val)
    )
    # Save VAE - Dense model 256
    #vae.save(directory)
    #vae.save_weights(vae_dense_256_checkpoint_path)         #buscar como se guarda el modelo 

    return model


