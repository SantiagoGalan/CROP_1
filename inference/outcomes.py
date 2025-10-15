import tensorflow as tf
import inference.metrics as met
import importlib

importlib.reload(met)

def outcomes(
    mask_source1,
    mask_source2,
    reconstructed_source1,
    reconstructed_source2,
    mixed_input,
    source1_gt,
    source2_gt,
    source1_cond,
    source2_cond,
    predictor,
):

    # Predictions from masks and reconstructions ---------------------------------
    y_predicted_s1_mask = predictor.predict(mask_source1, verbose=0)
    y_predicted_s2_mask = predictor.predict(mask_source2, verbose=0)

    y_predicted_s1_recon = predictor.predict(reconstructed_source1, verbose=0)
    y_predicted_s2_recon = predictor.predict(reconstructed_source2, verbose=0)

    y_predicted_mix = predictor.predict(reconstructed_source1, verbose=0)
    y_predicted_mix_orig = predictor.predict(mixed_input, verbose=0)

    y_predicted_gt1 = predictor.predict(source1_gt, verbose=0)

    y_reduced_gt1 = tf.math.argmax(source1_cond, 1)
    y_reduced_gt2 = tf.math.argmax(source2_cond, 1)

    # Best MSE-based selection ---------------------------------------------------
    select_s1 = tf.cast(
        tf.math.less(
            tf.keras.metrics.MSE(source1_gt, mask_source1),
            tf.keras.metrics.MSE(source2_gt, mask_source1),
        ),
        tf.float32,
    )
    select_s2 = tf.cast(
        tf.math.greater_equal(
            tf.keras.metrics.MSE(source1_gt, mask_source1),
            tf.keras.metrics.MSE(source2_gt, mask_source1),
        ),
        tf.float32,
    )
    select_s1 = tf.expand_dims(select_s1, 1)
    select_s2 = tf.expand_dims(select_s2, 1)
    x_best_MSE = (source1_gt * select_s1) + (source2_gt * select_s2)

    # Class-based selection (first predicted digit) -------------------------------
    s_best_s1 = tf.cast(
        tf.math.greater_equal(y_reduced_gt1, y_reduced_gt2), tf.int64
    )
    s_1_best_s1 = tf.cast(tf.math.less(y_reduced_gt1, y_reduced_gt2), tf.int64)

    y_best_predicted_1 = s_best_s1 * y_reduced_gt1 + s_1_best_s1 * y_reduced_gt2

    select_s1 = tf.cast(s_best_s1, tf.float32)
    select_s2 = tf.cast(s_1_best_s1, tf.float32)
    select_s1 = tf.expand_dims(select_s1, 1)
    select_s2 = tf.expand_dims(select_s2, 1)

    best_prediction_source1 = (source1_gt * select_s1) + (source2_gt * select_s2)

    # Metrics --------------------------------------------------------------------
    bpsnr = met.batched_psnr(
        source1_gt, source2_gt, reconstructed_source1, reconstructed_source2
    )
    bpsnr_d = met.batched_psnr(
        source1_gt, source2_gt, mask_source1, mask_source2
    )

    # New accuracy metric --------------------------------------------------------
    acc_at_least_one, acc_both = met.accuracys(
        p1=y_predicted_s1_recon,
        p2=y_predicted_s2_recon,
        y1=source1_cond,
        y2=source2_cond,
    )

    return (
        best_prediction_source1,
        y_predicted_s1_recon,
        y_predicted_s2_recon,
        bpsnr,
        bpsnr_d,
        acc_at_least_one,
        acc_both,
    )
