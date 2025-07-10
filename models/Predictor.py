from keras.layers import Input, Dense, Conv2D, BatchNormalization, Dropout, Flatten
from keras.models import Model
from custom_layers.ReshapeLayer import ReshapeLayer

def build_predictor():
    image_size = 28
    original_dim_C = (image_size, image_size, 1)                                                       # ¿agregar condición?
    original_dim = image_size * image_size

    n_cond = 10

    # Define the input layer
    # Define predictor model ---------------------------------------------------------
    input_predictor = Input(shape=(original_dim,), name="original_input")

    # Use the custom reshape layer
    input_predictor_C = ReshapeLayer(original_dim_C)(input_predictor)

    predictor_inputs = input_predictor_C
    '''
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(predictor_inputs)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    predictor_outputs = Dense(n_cond, activation="softmax")(x)
    '''
    x = Conv2D(32, 3, activation="relu", strides=2, padding="same")(predictor_inputs)
    x = BatchNormalization()(x)  # Add batch normalization
    x = Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = BatchNormalization()(x)  # Add batch normalization
    x = Flatten()(x)
    x = Dropout(0.5)(x)  # Add dropout
    predictor_outputs = Dense(n_cond, activation="softmax")(x)

    # instantiate decoder model

    predictor = Model(inputs=input_predictor, outputs=predictor_outputs, name="predictor")
    predictor.summary()
    return predictor