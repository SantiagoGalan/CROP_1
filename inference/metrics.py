from custom_layers.ReshapeLayer import ReshapeLayer
import tensorflow as tf
import numpy as np

def ssim_grayscale(target, preds):

    # Convert NumPy arrays to TensorFlow tensors
    target_tensor = tf.convert_to_tensor(target, dtype=tf.float32)
    preds_tensor = tf.convert_to_tensor(preds, dtype=tf.float32)

    # Add a batch dimension if the tensors are 1D
    if len(target_tensor.shape) == 1:
        target_tensor = tf.expand_dims(target_tensor, axis=0)
    if len(preds_tensor.shape) == 1:
        preds_tensor = tf.expand_dims(preds_tensor, axis=0)

    # Add a channel dimension to represent grayscale images
    target_tensor = tf.expand_dims(target_tensor, axis=-1)
    preds_tensor = tf.expand_dims(preds_tensor, axis=-1)

    original_dim_C = (28, 28, 1)

    # Use the custom reshape layer
    target_tensor_C = ReshapeLayer(original_dim_C)(target_tensor)
    preds_tensor_C = ReshapeLayer(original_dim_C)(preds_tensor)
    target_tensor = target_tensor_C
    preds_tensor = preds_tensor_C

    # Calculate SSIM
    ssim = tf.image.ssim(target_tensor, preds_tensor, max_val=1.0, filter_size=11,
                         filter_sigma=1.5, k1=0.01, k2=0.03)

    #print(ssim)
    return ssim


def batched_ssim(gt1, gt2, gen1, gen2):

    # Calculate SSIM for pairs (gt1, gen1) and (gt2, gen2)
    batched_ssim_12 = (
        0.5 * ssim_grayscale(gt1, gen1) +
        0.5 * ssim_grayscale(gt2, gen2)
    )

    #print(batched_ssim_12)

    # Calculate SSIM for pairs (gt1, gen2) and (gt2, gen1)
    batched_ssim_21 = (
        0.5 * ssim_grayscale(gt1, gen2) +
        0.5 * ssim_grayscale(gt2, gen1)
    )

    #print(batched_ssim_21)

    # Compute the maximum PSNR between the two pairs
    bssim_max = tf.math.maximum(batched_ssim_12, batched_ssim_21)

    # Compute the mean of the maximum PSNR values
    bssim_mean = tf.reduce_mean(bssim_max, axis=None, keepdims=False)

    # Compute the standard deviation of the maximum PSNR values
    bssim_std = tf.math.reduce_std(bssim_max, axis=None, keepdims=False)

    return bssim_mean, bssim_std


def psnr_grayscale(target, preds):

    # Convert NumPy arrays to TensorFlow tensors
    target_tensor = tf.convert_to_tensor(target, dtype=tf.float32)
    preds_tensor = tf.convert_to_tensor(preds, dtype=tf.float32)

    # Add a batch dimension if the tensors are 1D
    if len(target_tensor.shape) == 1:
        target_tensor = tf.expand_dims(target_tensor, axis=0)
    if len(preds_tensor.shape) == 1:
        preds_tensor = tf.expand_dims(preds_tensor, axis=0)

    # Add a channel dimension to represent grayscale images
    target_tensor = tf.expand_dims(target_tensor, axis=-1)
    preds_tensor = tf.expand_dims(preds_tensor, axis=-1)

    original_dim_C = (28, 28, 1)

    # Use the custom reshape layer
    target_tensor_C = ReshapeLayer(original_dim_C)(target_tensor)
    preds_tensor_C = ReshapeLayer(original_dim_C)(preds_tensor)
    target_tensor = target_tensor_C
    preds_tensor = preds_tensor_C

    # Calculate PSNR
    psnr = tf.image.psnr(target_tensor, preds_tensor, max_val=1.0)
    #print(psnr)
    return psnr

def batched_psnr(gt1, gt2, gen1, gen2):

    # Calculate PSNR for pairs (gt1, gen1) and (gt2, gen2)
    batched_psnr_12 = (
        0.5 * psnr_grayscale(gt1, gen1) +
        0.5 * psnr_grayscale(gt2, gen2)
    )

#    print(batched_psnr_12)

    # Calculate PSNR for pairs (gt1, gen2) and (gt2, gen1)
    batched_psnr_21 = (
        0.5 * psnr_grayscale(gt1, gen2) +
        0.5 * psnr_grayscale(gt2, gen1)
    )

#    print(batched_psnr_21)

    # Compute the maximum PSNR between the two pairs
    bpsnr_max = tf.math.maximum(batched_psnr_12, batched_psnr_21)

    # Compute the mean of the maximum PSNR values
    bpsnr_mean = tf.reduce_mean(bpsnr_max, axis=None, keepdims=False)

    # Compute the standard deviation of the maximum PSNR values
    bpsnr_std = tf.math.reduce_std(bpsnr_max, axis=None, keepdims=False)

    return bpsnr_mean, bpsnr_std


def accuracys(p1,p2,y1,y2):
    p1_max = np.argmax(p1, axis=1)
    p2_max = np.argmax(p2, axis=1)
    y1_max = np.argmax(y1, axis=1)
    y2_max = np.argmax(y2, axis=1)

    at_least_one = ((p1_max == y1_max) |
           (p1_max == y2_max) |
           (p2_max == y1_max) |
           (p2_max == y2_max)).astype(int)

    pred_pairs = np.sort(np.stack([p1_max, p2_max], axis=1), axis=1)
    y_pairs = np.sort(np.stack([y1_max, y2_max], axis=1), axis=1)

    # Comparamos fila a fila
    both = np.all(pred_pairs == y_pairs, axis=1).astype(int)

    acc_at_least_one = np.count_nonzero(at_least_one)/len(at_least_one)
    acc_both =  np.count_nonzero(both)/len(both)

    return acc_at_least_one, acc_both