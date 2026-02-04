from keras.layers import Input, Dense, Concatenate
from keras.models import Model
from project.custom_layers.sampling import Sampling


class Encoder(Model):
    def __init__(
        self,
        img_dim=(28, 28),
        condition_dim=(10,),
        intermediate_dim=128,
        latent_dim=2,
        name="encoder",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        flat_dim = img_dim[0] * img_dim[1]

        # Inputs
        img_input = Input(shape=(flat_dim,), name="img_input_encoder")
        cond_input = Input(shape=condition_dim, name="encoder_condition")

        # Architecture
        x = Concatenate()([img_input, cond_input])
        x = Dense(intermediate_dim, activation="relu")(x)

        z_mean = Dense(latent_dim, name="z_mean")(x)
        z_log_var = Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()((z_mean, z_log_var))

        # Internal functional model
        self._model = Model(
            inputs=[img_input, cond_input],
            outputs=[z_mean, z_log_var, z],
            name=name
        )

    def call(self, inputs, training=False):
        return self._model(inputs, training=training)
