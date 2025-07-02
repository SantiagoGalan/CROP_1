from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def train(model, x_train, y_train, x_val, y_val, epochs=20, batch_size=128, output_path="vae_best_model.h5"):
    """
    Entrena el modelo VAE.

    Parámetros:
    -----------
    model : keras.Model
        Modelo VAE ya instanciado y listo para compilar.
    x_train, y_train : np.array
        Datos de entrenamiento (inputs y condiciones).
    x_val, y_val : np.array
        Datos de validación.
    epochs : int
        Número de épocas a entrenar.
    batch_size : int
        Tamaño del batch.
    output_path : str
        Ruta para guardar el mejor modelo.

    Retorna:
    --------
    history : keras.callbacks.History
        Historial del entrenamiento.
    """

    # Compilar el modelo aquí (puedes cambiar optimizador si querés)
    model.compile(optimizer='adam')

    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint(output_path, save_best_only=True, monitor='val_loss')

    history = model.fit(
        x=[x_train, y_train, y_train],
        y=None,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=([x_val, y_val, y_val], None),   # ahora tres
        callbacks=[early_stop, checkpoint]
    )

    return history