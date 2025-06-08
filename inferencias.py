import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import outcomes
import PNSR
import SSIM
import importlib

importlib.reload(outcomes)

def best_digit_var_sigmoid(x_mix_filtrado_2, x_mix_orig, alpha, bias, slope, predictor, encoder, decoder):
    """
    Filtra y decodifica una imagen mezclada usando el encoder y decoder, aplicando un ajuste con parámetros alpha, bias y slope.
    """
    x_mix_filtrado_1 = (2 * x_mix_orig - x_mix_filtrado_2)
    x_mix_filtrado_1 = tf.clip_by_value(x_mix_filtrado_1, clip_value_min=0, clip_value_max=1)

    if x_mix_filtrado_1.ndim == 2:
        x_mix_filtrado_1 = np.expand_dims(x_mix_filtrado_1, axis=0)

    condition_encoder = predictor.predict(x_mix_filtrado_1)
    condition_decoder_1 = condition_encoder

    latent_inputs = encoder.predict([x_mix_filtrado_1, condition_encoder], verbose=0)
    x_decoded_1 = decoder.predict([latent_inputs[2], condition_decoder_1], verbose=0)

    x_decoded_1 = (x_decoded_1 - bias) * slope
    x_decoded_1 = tf.sigmoid(x_decoded_1)
    x_decoded_1 = np.squeeze(x_decoded_1)

    x_mix_filtrado_1 = (2 * x_mix_orig * x_decoded_1)
    x_mix_filtrado_1 = tf.clip_by_value(x_mix_filtrado_1, clip_value_min=0, clip_value_max=1)

    return (x_mix_filtrado_1, x_decoded_1)

def mostrar_imagenes(titulo, imagenes, etiquetas=None, n=5):
    """
    Visualiza un conjunto de imágenes con sus etiquetas.
    """
    plt.figure(figsize=(n * 2, 2))
    for i in range(n):
        img = imagenes[i]
        if hasattr(img, "numpy"):
            img = img.numpy()
        plt.subplot(1, n, i + 1)
        plt.imshow(img.reshape(28, 28), cmap='gray')
        if etiquetas is not None:
            plt.title(str(etiquetas[i]))
        plt.axis('off')
    plt.suptitle(titulo)
    plt.show()

def inferncia_modelo(x_train, x_train_1, y_train, predictor, encoder, decoder, y_train_1):
    """
    Realiza inferencias iterativas sobre imágenes mezcladas, filtrando y decodificando en cada paso,
    y calcula métricas de calidad (PSNR y SSIM) entre las imágenes originales y generadas.
    Visualiza imágenes originales, separadas y resultados intermedios.
    """
    alfa_mix = 0.5
    average_image = alfa_mix * x_train.astype(np.float32) + (1 - alfa_mix) * x_train_1.astype(np.float32)
    x_train_mix = average_image

    # Inicialización
    x_train_mix_orig = x_train_mix
    x_train_decoded_1 = x_train_mix
    x_train_decoded_2 = x_train_mix
    x_train_mix_filtrado_1 = x_train_mix
    x_train_mix_filtrado_2 = x_train_mix
    x__x = tf.zeros_like(x_train_mix)

    Iterations = 3
    bias = 0.22
    slope = 22.
    beta = 1.
    alpha_1 = -2
    alpha_2 = -22

    # Visualiza las imágenes originales y la mezcla inicial
    mostrar_imagenes("Imagen original 1", x_train)
    mostrar_imagenes("Imagen original 2", x_train_1)
    mostrar_imagenes("Imagen mezclada inicial", x_train_mix_orig)

    for j in range(Iterations):
        x_train_mix_filtrado_1, x_train_decoded_1 = best_digit_var_sigmoid(
            x_train_mix_filtrado_2, x_train_mix_orig, alpha_2, bias, slope, predictor, encoder, decoder)
        alpha_2 = alpha_2 * beta

        x_train_mix_filtrado_2, x_train_decoded_2 = best_digit_var_sigmoid(
            x_train_mix_filtrado_1, x_train_mix_orig, alpha_1, bias, slope, predictor, encoder, decoder)
        alpha_1 = alpha_1 * beta

        print(f"ITERACIÓN A: {j}")

        # Visualiza resultados intermedios de la inferencia
        mostrar_imagenes(f"Iteración {j+1} - x_mix_filtrado_1", x_train_mix_filtrado_1)
        mostrar_imagenes(f"Iteración {j+1} - x_mix_filtrado_2", x_train_mix_filtrado_2)
        mostrar_imagenes(f"Iteración {j+1} - x_decoded_1", x_train_decoded_1)
        mostrar_imagenes(f"Iteración {j+1} - x_decoded_2", x_train_decoded_2)

        # Evaluación de resultados
        _, y_train_predicted_1_f, y_train_predicted_2_f = outcomes.outcomes(
            x_train_decoded_1, x_train_decoded_2, x_train_mix_filtrado_1,
            x_train_mix_filtrado_2, x_train_mix_orig, x_train, x_train_1,
            y_train, y_train_1, predictor)

        # Métricas de calidad de imagen
        gt1 = x_train
        gt2 = x_train_1
        gen1 = x_train_mix_filtrado_1
        gen2 = x_train_mix_filtrado_2

        # PSNR
        bpsnr_mean, bpsnr_std = PNSR.batched_psnr(gt1, gt2, gen1, gen2)
        print("Batched PSNR Mean:", bpsnr_mean.numpy())
        print("Batched PSNR Std:", bpsnr_std.numpy())

        # SSIM
        bssim_mean, bssim_std = SSIM.batched_ssim(gt1, gt2, gen1, gen2)
        print("Batched SSIM Mean:", bssim_mean.numpy())
        print("Batched SSIM Std:", bssim_std.numpy())

    # Visualiza las imágenes separadas finales
    mostrar_imagenes("Imagen separada final 1 (x_decoded_1)", x_train_decoded_1)
    mostrar_imagenes("Imagen separada final 2 (x_decoded_2)", x_train_decoded_2)