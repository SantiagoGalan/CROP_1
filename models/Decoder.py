from keras.layers import Input, Dense, Concatenate, Reshape
from keras.models import Model

def build_decoder(latent_dim=2, cond_dim=(10,), intermediate_dim=128, original_shape=(28, 28)):
    original_dim = original_shape[0] * original_shape[1]

    z_inputs = Input(shape=(latent_dim,), name="z_sampling")
    cond_decoder = Input(shape=cond_dim, name="decoder_condition")
    
    latent_inputs = Concatenate()([z_inputs, cond_decoder])

    x = Dense(intermediate_dim, activation="relu")(latent_inputs)
    x = Dense(original_dim, activation="sigmoid")(x)

    # Reshape a imagen 2D
    decoder_outputs = Reshape(original_shape)(x)

    decoder = Model(inputs=[z_inputs, cond_decoder], outputs=decoder_outputs, name="decoder")
    return decoder
