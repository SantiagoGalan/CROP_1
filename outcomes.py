import numpy as np
import tensorflow as tf
import PNSR
import SSIM

def outcomes(
    x_decoded_1, x_decoded_2, x_mix_filtrado_1, x_mix_filtrado_2, x_mix_orig,
    x, x_1, y, y_1, predictor
):
    """
    Evalúa los resultados de la reconstrucción y filtrado de imágenes.

    Parámetros:
    -----------
    x_decoded_1, x_decoded_2 : np.ndarray
        Imágenes decodificadas por el modelo.
    x_mix_filtrado_1, x_mix_filtrado_2 : np.ndarray
        Imágenes filtradas.
    x_mix_orig : np.ndarray
        Imagen mezclada original.
    x, x_1 : np.ndarray
        Imágenes originales.
    y, y_1 : np.ndarray
        Etiquetas originales (one-hot o enteros).
    predictor : modelo keras
        Modelo para predecir la condición.

    Retorna:
    --------
    x_best_predicted_1 : np.ndarray
        Mejor imagen según predicción.
    y_predicted_1_f, y_predicted_2_f : np.ndarray
        Predicciones de las imágenes filtradas.
    """

    # Chequeo de dimensiones
    n = x.shape[0]
    
    for arr in [x_decoded_1, x_decoded_2, x_mix_filtrado_1, x_mix_filtrado_2, x_mix_orig, x_1, y, y_1]:
        for a in arr:
            print(a.shape)
        if arr.shape[0] != n:
            raise ValueError(f"Todos los arrays deben tener la misma cantidad de muestras. Esperado {n}, recibido {arr.shape[0]}.")

    # Predicciones
    y_predicted_1_f = tf.math.argmax(predictor.predict(x_mix_filtrado_1), 1)
    y_predicted_2_f = tf.math.argmax(predictor.predict(x_mix_filtrado_2), 1)
    y_predicted_mix_orig = tf.math.argmax(predictor.predict(x_mix_orig), 1)
    y_predicted = tf.math.argmax(predictor.predict(x), 1)
    y_predicted_1 = tf.math.argmax(predictor.predict(x_1), 1)
    y_reduced = tf.math.argmax(y, 1) if y.ndim > 1 else y
    y_1_reduced = tf.math.argmax(y_1, 1) if y_1.ndim > 1 else y_1

    # Selección de la mejor imagen según MSE

    
    # Calcula el MSE por muestra
    mse_x = tf.reduce_mean(tf.square(tf.reshape(x, (n, -1)) - tf.reshape(x_decoded_1, (n, -1))), axis=1)
    mse_x1 = tf.reduce_mean(tf.square(tf.reshape(x_1, (n, -1)) - tf.reshape(x_decoded_1, (n, -1))), axis=1)

    # Asegura que select sea un vector
    select = tf.cast(mse_x < mse_x1, tf.float32)
    select_1 = 1.0 - select

    # Solo expande si tiene al menos 1 dimensión
    # Expande a [n, 1, 1] para que sea compatible con [n, 28, 28]
    select = tf.reshape(select, (-1, 1, 1))
    select_1 = tf.reshape(select_1, (-1, 1, 1))

    x_best_MSE = (x * select) + (x_1 * select_1)
    
    # Métricas de calidad
    bpsnr_mean, bpsnr_std = PNSR.batched_psnr(x, x_1, x_mix_filtrado_1, x_mix_filtrado_2)
    bssim_mean, bssim_std = SSIM.batched_ssim(x, x_1, x_mix_filtrado_1, x_mix_filtrado_2)
    print("PSNR mean:", bpsnr_mean.numpy(), "SSIM mean:", bssim_mean.numpy())

    # Devuelve la mejor imagen y las predicciones filtradas
    return x_best_MSE, y_predicted_1_f, y_predicted_2_f