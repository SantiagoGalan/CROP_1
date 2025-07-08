from keras.models import Model
import tensorflow as tf

class CVAE(Model):
    def __init__(self, encoder, decoder, original_dim, beta=1.0):
        super(CVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.original_dim = original_dim
        self.beta = beta
        self.mse = tf.keras.losses.MeanSquaredError()

    def call(self, inputs):
        x, c = inputs
        z, z_mean, z_log_var = self.encoder([x, c])
        reconstruction = self.decoder([z, c])

        # Guardamos para el cálculo de la pérdida
        self.z_mean = z_mean
        self.z_log_var = z_log_var
        self.reconstruction = reconstruction

        return reconstruction

    def train_step(self, data):
        (x, c), y_true = data

        with tf.GradientTape() as tape:
            y_pred = self([x, c], training=True)
            reconstruction_loss = self.mse(tf.reshape(y_true, [-1, self.original_dim]),
                                           tf.reshape(y_pred, [-1, self.original_dim])) * self.original_dim
            kl_loss = -0.5 * tf.reduce_sum(
                1 + self.z_log_var - tf.square(self.z_mean) - tf.exp(self.z_log_var), axis=1)
            kl_loss = tf.reduce_mean(kl_loss)
            total_loss = reconstruction_loss + self.beta * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss
        }
