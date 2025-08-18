import tensorflow as tf

class CVAE(tf.keras.Model):
    def __init__(self, encoder, decoder, original_dim, beta=1.0,**kwargs):
            super(CVAE, self).__init__(**kwargs)
            self.encoder = encoder
            self.decoder = decoder
            self.original_dim = original_dim
            self.beta = beta
            self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
            self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
            self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")


    def call(self, inputs):
        x, cond = inputs
        _, _, z = self.encoder([x, cond])
        return self.decoder([z, cond])

    def train_step(self, data):
        (inputs, targets) = data
        x, cond = inputs

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder([x, cond], training=True)
            reconstruction = self.decoder([z, cond], training=True)
            
            reconstruction_loss = tf.keras.losses.mse(tf.reshape(x, [-1, self.original_dim]),
                                    tf.reshape(reconstruction, [-1, self.original_dim]))* self.original_dim #???

            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_sum(kl_loss,-1)

            total_loss = reconstruction_loss + self.beta * kl_loss
            total_loss = tf.reduce_mean(total_loss)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        (inputs, targets) = data
        x, cond = inputs

        z_mean, z_log_var, z = self.encoder([x, cond], training=False)
        reconstruction = self.decoder([z, cond], training=False)

        reconstruction_loss = tf.keras.losses.mse(
            tf.reshape(x, [-1, self.original_dim]),
            tf.reshape(reconstruction, [-1, self.original_dim])
        ) * self.original_dim

        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_sum(kl_loss, axis=-1)

        total_loss = tf.reduce_mean(reconstruction_loss + self.beta * kl_loss)

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
