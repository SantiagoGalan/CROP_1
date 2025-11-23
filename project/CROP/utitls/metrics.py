"""
Modulo de métricas de CROP
"""

import numpy as np

from project.custom_layers.reshapeLayer import ReshapeLayer
import tensorflow as tf
import numpy as np

class Metrics():
    
    @classmethod
    def ssim_grayscale(cls,target, preds):

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
        ssim = tf.image.ssim(
            target_tensor,
            preds_tensor,
            max_val=1.0,
            filter_size=11,
            filter_sigma=1.5,
            k1=0.01,
            k2=0.03,
        )

        # print(ssim)
        return ssim

    @classmethod
    def batched_ssim(cls,gt1, gt2, gen1, gen2):

        # Calculate SSIM for pairs (gt1, gen1) and (gt2, gen2)
        batched_ssim_12 = 0.5 * cls.ssim_grayscale(gt1, gen1) + 0.5 * cls.ssim_grayscale(gt2, gen2)

        # print(batched_ssim_12)

        # Calculate SSIM for pairs (gt1, gen2) and (gt2, gen1)
        batched_ssim_21 = 0.5 * cls.ssim_grayscale(gt1, gen2) + 0.5 * cls.ssim_grayscale(gt2, gen1)

        # print(batched_ssim_21)

        # Compute the maximum PSNR between the two pairs
        bssim_max = tf.math.maximum(batched_ssim_12, batched_ssim_21)

        # Compute the mean of the maximum PSNR values
        bssim_mean = tf.reduce_mean(bssim_max, axis=None, keepdims=False)

        # Compute the standard deviation of the maximum PSNR values
        bssim_std = tf.math.reduce_std(bssim_max, axis=None, keepdims=False)

        return bssim_mean, bssim_std

    @classmethod
    def psnr_grayscale(cls,target, preds):

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
        # print(psnr)
        return psnr

    @classmethod
    def batched_psnr(cls,gt1, gt2, gen1, gen2):

        # Calculate PSNR for pairs (gt1, gen1) and (gt2, gen2)
        batched_psnr_12 = 0.5 * cls.psnr_grayscale(gt1, gen1) + 0.5 * cls.psnr_grayscale(gt2, gen2)

        #    print(batched_psnr_12)

        # Calculate PSNR for pairs (gt1, gen2) and (gt2, gen1)
        batched_psnr_21 = 0.5 * cls.psnr_grayscale(gt1, gen2) + 0.5 * cls.psnr_grayscale(gt2, gen1)

        #    print(batched_psnr_21)

        # Compute the maximum PSNR between the two pairs
        bpsnr_max = tf.math.maximum(batched_psnr_12, batched_psnr_21)

        # Compute the mean of the maximum PSNR values
        bpsnr_mean = tf.reduce_mean(bpsnr_max, axis=None, keepdims=False)

        # Compute the standard deviation of the maximum PSNR values
        bpsnr_std = tf.math.reduce_std(bpsnr_max, axis=None, keepdims=False)

        return bpsnr_mean, bpsnr_std

    @classmethod
    def accuracys(cls,gt1, gt2, p1, p2):
        gt1_max = np.argmax(gt1, axis=1)
        gt2_max = np.argmax(gt2, axis=1)
        p1_max = np.argmax(p1, axis=1)
        p2_max = np.argmax(p2, axis=1)

        at_least_one = (
            (p1_max == gt1_max)
            | (p1_max == gt2_max)
            | (p2_max == gt1_max)
            | (p2_max == gt2_max)
        ).astype(int)

        pred_pairs = np.sort(np.stack([p1_max, p2_max], axis=1), axis=1)
        y_pairs = np.sort(np.stack([gt1_max, gt2_max], axis=1), axis=1)
        both = np.all(pred_pairs == y_pairs, axis=1).astype(int)

        acc_at_least_one = np.count_nonzero(at_least_one) / len(at_least_one)
        acc_both = np.count_nonzero(both) / len(both)

        return round(acc_at_least_one, 2), round(acc_both, 2)
    
    @classmethod
    def best_predicctions(cls,gt1,gt2,source1_cond,
    source2_cond):

        y_reduced_gt1 = tf.math.argmax(source1_cond, 1)
        y_reduced_gt2 = tf.math.argmax(source2_cond, 1)


        s_best_s1 = tf.cast(tf.math.greater_equal(y_reduced_gt1, y_reduced_gt2), tf.int64)
        s_1_best_s1 = tf.cast(tf.math.less(y_reduced_gt1, y_reduced_gt2), tf.int64)

        select_s1 = tf.cast(s_best_s1, tf.float32)
        select_s2 = tf.cast(s_1_best_s1, tf.float32)
        select_s1 = tf.expand_dims(select_s1, 1)
        select_s2 = tf.expand_dims(select_s2, 1)

        return (gt1 * select_s1) + (gt2 * select_s2)
 
