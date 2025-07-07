from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Lambda
import tensorflow as tf


## chequear el procesamiento de los datosen esta version.
## chequear la entrada de estos modelos. 

def build_vae(encoder, decoder,x_train,y_train,x_val,y_val,
              img_shape=(28, 28, 1), cond_dim=10, beta=1.0,epochs=5,
    batch_size=128):
    """
    VAE funcional que recibe:
    - imagen
    - condición_encoder
    - condición_decoder
    e incorpora pérdida KL con add_loss simbólica dentro del grafo.

    # KL loss dentro de una capa Lambda que no modifica flujo, pero agrega pérdida
    def kl_with_loss(args):
        z_mean, z_log_var = args
        kl = -0.5 * tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1
        )
        
        kl = tf.reduce_mean(kl)
        tf.keras.backend.add_loss(beta * kl)
        return z_mean  # devuelve algo arbitrario

    _ = Lambda(kl_with_loss, output_shape=(None, z_mean.shape[-1]))([z_mean, z_log_var])

    """
    # Entradas
    img_input = Input(shape=img_shape, name="image_input")
    cond_enc_input = Input(shape=(cond_dim,), name="encoder_condition")
    cond_dec_input = Input(shape=(cond_dim,), name="decoder_condition")

    # Encoder
    z_mean, z_log_var, z = encoder([img_input, cond_enc_input])

# Decoder
    x_rec = decoder([z, cond_dec_input])

    # Modelo final
    vae_model = Model(inputs=[img_input, cond_enc_input, cond_dec_input],
                      outputs=x_rec,
                      name="vae")

    # entrenamiento del modelo
    

    # Loss -----------------------------------------------------------------------
    def vae_loss(y_true, y_pred):
            
        kl_loss = -0.5 * (z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        kl_loss = tf.reduce_sum(kl_loss, -1)

        mse = tf.keras.losses.MeanSquaredError()
        mse1 = mse(y_true, y_pred) * (28*28) #original_dim 

            # Add KL divergence regularization loss.
        loss = mse1 + beta * kl_loss
        return tf.reduce_mean(loss) #, axis=-1)

    # Optimizer ------------------------------------------------------------------
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    # Compile --------------------------------------------------------------------
    vae_model.compile(optimizer, loss=vae_loss)
## si no anda agregar los datos y entrenamineto aca....

    vae_model.fit(
        x=[x_train,y_train,y_train],
        y=[x_train],
        validation_data=([x_val,y_val,y_val],x_val),
        epochs=epochs,
        batch_size=batch_size
    )

    return vae_model
