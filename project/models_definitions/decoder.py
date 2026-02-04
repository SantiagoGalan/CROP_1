from keras.layers import Input, Dense, Concatenate
from keras.models import Model


class Decoder(Model):
    def __init__(
        self,
        latent_dim=2,
        cond_dim=(10,),
        intermediate_dim=128,
        original_shape=(28, 28),
        name="decoder",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self.original_dim = original_shape[0] * original_shape[1]

        # Inputs
        z_inputs = Input(shape=(latent_dim,), name="z_sampling")
        cond_inputs = Input(shape=cond_dim, name="decoder_condition")

        # Architecture
        x = Concatenate()([z_inputs, cond_inputs])
        x = Dense(intermediate_dim, activation="relu")(x)
        outputs = Dense(self.original_dim, activation="sigmoid")(x)

        # Build functional model internally
        self._model = Model(
            inputs=[z_inputs, cond_inputs],
            outputs=outputs,
            name=name
        )

    def call(self, inputs, training=False):
        return self._model(inputs, training=training)
