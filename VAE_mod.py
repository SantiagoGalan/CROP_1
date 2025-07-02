import tensorflow as tf
from tensorflow.keras import Model

class VAE(Model):
    def __init__(self, encoder, decoder, show_model=False):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam()

        # Métricas
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

        if show_model:
            from keras.utils import plot_model
            from IPython.display import Image, display
            plot_model(self.encoder, to_file="encoder.png", show_shapes=True, show_layer_names=True)
            display(Image(filename="encoder.png"))
            plot_model(self.decoder, to_file="decoder.png", show_shapes=True, show_layer_names=True)
            display(Image(filename="decoder.png"))

    @property
    def metrics(self):
        # Keras reseteará automáticamente estas métricas al iniciar cada epoch
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, inputs):
        imagen, condicion_encoder, condicion_decoder = inputs
        z_mean, z_log_var, z = self.encoder([imagen, condicion_encoder])
        reconstruccion = self.decoder([z, condicion_decoder])
        return reconstruccion

    def train_step(self, data):
        # Desempaquetar los inputs (data == ([x, cond_enc, cond_dec], None))
        if isinstance(data, tuple):
            inputs = data[0]
        else:
            inputs = data
        x, cond_enc, cond_dec = inputs

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder([x, cond_enc], training=True)
            reconstruction = self.decoder([z, cond_dec], training=True)

            # Pérdida de reconstrucción (BCE)
            reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(x, reconstruction)
            ) * tf.cast(tf.reduce_prod(tf.shape(x)[1:]), tf.float32)

            # Pérdida KL
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )

            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Actualizar métricas
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
            if isinstance(data, tuple):
                inputs = data[0]
            else:
                inputs = data
            x, cond_enc, cond_dec = inputs

            # Forward pass (sin GradientTape)
            z_mean, z_log_var, z = self.encoder([x, cond_enc], training=False)
            reconstruction = self.decoder([z, cond_dec], training=False)

            # Mismatched shapes ya corregidos (x tiene canal 1)
            reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(x, reconstruction)
            ) * tf.cast(tf.reduce_prod(tf.shape(x)[1:]), tf.float32)
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            total_loss = reconstruction_loss + kl_loss

            # Actualizar métricas
            self.total_loss_tracker.update_state(total_loss)
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            self.kl_loss_tracker.update_state(kl_loss)

            return {
                "loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result(),
            }