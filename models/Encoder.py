from keras.layers import Input, Concatenate, Dense, Flatten
from keras.models import Model
from custom_layers.Sampling import Sampling


def build_enconder(img_dim=(28,28),condition_dim=(10,),intermediate_dim=128,latent_dim=2):
    flat_dim = img_dim[0]*img_dim[1]
    
    img_input = Input(shape=(flat_dim,), name="img_input_encoder")
    cond_encoder = Input(shape=(condition_dim), name="encoder_condition")

    imputs_cocanteados = Concatenate()([img_input, cond_encoder])
    
    x = Dense(intermediate_dim, activation="relu")(imputs_cocanteados)
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()((z_mean, z_log_var))
    encoder = Model(inputs=[img_input, cond_encoder], outputs=[z_mean, z_log_var, z], name="encoder")
    return encoder