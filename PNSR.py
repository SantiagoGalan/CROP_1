import numpy as np
import tensorflow as tf
import layers.ReshapeLayer as ReshapeLayer

def psnr_grayscale(target, preds):
    """
    Calcula el PSNR (Peak Signal-to-Noise Ratio) entre dos imágenes en escala de grises.
    """
    target_tensor = tf.convert_to_tensor(target, dtype=tf.float32)
    preds_tensor = tf.convert_to_tensor(preds, dtype=tf.float32)

    # Añade canal solo si es necesario
    if target_tensor.shape.ndims == 3:
        target_tensor = tf.expand_dims(target_tensor, axis=-1)
    if preds_tensor.shape.ndims == 3:
        preds_tensor = tf.expand_dims(preds_tensor, axis=-1)

    original_dim_C = (28, 28, 1)  # Considera parametrizar esto
    target_tensor = ReshapeLayer.ReshapeLayer(original_dim_C)(target_tensor)
    preds_tensor = ReshapeLayer.ReshapeLayer(original_dim_C)(preds_tensor)

    psnr = tf.image.psnr(target_tensor, preds_tensor, max_val=1.0)
    return psnr

def batched_psnr(gt1, gt2, gen1, gen2):
    """
    Calcula el PSNR promedio y su desviación estándar entre pares de imágenes reales y generadas.
    """
    batched_psnr_12 = 0.5 * psnr_grayscale(gt1, gen1) + 0.5 * psnr_grayscale(gt2, gen2)
    batched_psnr_21 = 0.5 * psnr_grayscale(gt1, gen2) + 0.5 * psnr_grayscale(gt2, gen1)
    bpsnr_max = tf.math.maximum(batched_psnr_12, batched_psnr_21)
    bpsnr_mean = tf.reduce_mean(bpsnr_max)
    bpsnr_std = tf.math.reduce_std(bpsnr_max)
    return bpsnr_mean, bpsnr_std

'''
gt1 = x_test
gt2 = x_test_1
gen1 = x_test_mix_filtrado_1
gen2 = x_test_mix_filtrado_2

# Calculate batched PSNR mean and standard deviation
bpsnr_mean, bpsnr_std = batched_psnr(gt1, gt2, gen1, gen2)

# Print the results
print("Batched PSNR Mean:", bpsnr_mean.numpy())
print("Batched PSNR Standard Deviation:", bpsnr_std.numpy())
'''