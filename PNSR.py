import numpy as np
import tensorflow as tf
from keras import layers
from keras.layers import  Concatenate
from keras.datasets import mnist
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
mse = MeanSquaredError()
binary_crossentropy = BinaryCrossentropy()
from keras import backend as K
import matplotlib as mplt
import matplotlib.pyplot as plt
import Predictor
import ReshapeLayer

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
    target_tensor_C = ReshapeLayer.ReshapeLayer(original_dim_C)(target_tensor)
    preds_tensor_C = ReshapeLayer.ReshapeLayer(original_dim_C)(preds_tensor)
    target_tensor = target_tensor_C
    preds_tensor = preds_tensor_C

    # Calculate PSNR
    psnr = tf.image.psnr(target_tensor, preds_tensor, max_val=1.0)
    print(psnr)
    return psnr

def batched_psnr(gt1, gt2, gen1, gen2):

    # Calculate PSNR for pairs (gt1, gen1) and (gt2, gen2)
    batched_psnr_12 = (
        0.5 * psnr_grayscale(gt1, gen1) +
        0.5 * psnr_grayscale(gt2, gen2)
    )

    print(batched_psnr_12)

    # Calculate PSNR for pairs (gt1, gen2) and (gt2, gen1)
    batched_psnr_21 = (
        0.5 * psnr_grayscale(gt1, gen2) +
        0.5 * psnr_grayscale(gt2, gen1)
    )

    print(batched_psnr_21)

    # Compute the maximum PSNR between the two pairs
    bpsnr_max = tf.math.maximum(batched_psnr_12, batched_psnr_21)

    # Compute the mean of the maximum PSNR values
    bpsnr_mean = tf.reduce_mean(bpsnr_max, axis=None, keepdims=False)

    # Compute the standard deviation of the maximum PSNR values
    bpsnr_std = tf.math.reduce_std(bpsnr_max, axis=None, keepdims=False)

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