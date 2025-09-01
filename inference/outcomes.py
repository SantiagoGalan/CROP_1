import tensorflow as tf
import inference.metrics as met
import importlib

importlib.reload(met)
# Function *******************************************************
# OUTCOMES LIMPIO----------------

"""
def outcomes(
    x_decoded_1,
    x_decoded_2,
    x_mix_filter_1,
    x_mix_filter_2,
    x_mix_orig,
    x,
    x_1,
    y,
    y_1,
    predictor,
):

    # Outcomes ---------------------------------------------------------------------
    y_predicted_1 = tf.math.argmax(predictor.predict(x_decoded_1, verbose=0), 1)
    y_predicted_2 = tf.math.argmax(predictor.predict(x_decoded_2, verbose=0), 1)
    y_predicted_1_f = tf.math.argmax(predictor.predict(x_mix_filter_1, verbose=0), 1)
    y_predicted_2_f = tf.math.argmax(predictor.predict(x_mix_filter_2, verbose=0), 1)
    y_predicted_mix = tf.math.argmax(predictor.predict(x_mix_filter_1, verbose=0), 1)
    y_predicted_mix_orig = tf.math.argmax(predictor.predict(x_mix_orig, verbose=0), 1)
    y_predicted = tf.math.argmax(predictor.predict(x, verbose=0), 1)
    #y_predicted_1 = tf.math.argmax(predictor.predict(x_1, verbose=0), 1)
    y_reduced = tf.math.argmax(y, 1)
    y_1_reduced = tf.math.argmax(y_1, 1)

    select = tf.cast(
        tf.math.less(
            tf.keras.metrics.MSE(x, x_decoded_1), tf.keras.metrics.MSE(x_1, x_decoded_1)
        ),
        tf.float32,
    )
    select_1 = tf.cast(
        tf.math.greater_equal(
            tf.keras.metrics.MSE(x, x_decoded_1), tf.keras.metrics.MSE(x_1, x_decoded_1)
        ),
        tf.float32,
    )
    select = tf.expand_dims(select, 1)
    select_1 = tf.expand_dims(select_1, 1)
    x_best_MSE = (x * select) + (x_1 * select_1)

    select_1 = tf.cast(
        tf.math.equal(y_reduced, y_predicted_1_f), tf.int64
    )  # y   = y_predicted_1_f
    select_1_1 = tf.cast(
        tf.math.equal(y_1_reduced, y_predicted_1_f), tf.int64
    )  # y_1 = y_predicted_1_f
    select_2 = tf.cast(
        tf.math.equal(y_reduced, y_predicted_2_f), tf.int64
    )  # y   = y_predicted_2_f
    select_2_1 = tf.cast(
        tf.math.equal(y_1_reduced, y_predicted_2_f), tf.int64
    )  # y_1 = y_predicted_2_f

    y_s1 = y_reduced * select_1
    y_1_s1 = y_1_reduced * select_1_1
    y_s2 = y_reduced * select_2
    y_1_s2 = y_1_reduced * select_2_1

    select_1_AND = select_1 * select_1_1
    select_1_OR = select_1 + select_1_1 - select_1_AND
    select_2_AND = select_2 * select_2_1
    select_2_OR = select_2 + select_2_1 - select_2_AND
    s_best_AND = select_1_OR * select_2_OR
    s_best_OR = select_1_OR + select_2_OR - s_best_AND
    select_reduced = tf.cast(tf.math.not_equal(y_reduced, y_1_reduced), tf.int64)
    select_12_f = tf.cast(tf.math.equal(y_predicted_1_f, y_predicted_2_f), tf.int64)
    s_reduced_12_f_AND = select_reduced * select_12_f
    s_best_AND_AND = s_best_AND - s_reduced_12_f_AND

    # First predicted digit is equal to any of the original two digits --------------------------
    s_best_s1 = tf.cast(tf.math.greater_equal(y_s1, y_1_s1), tf.int64)
    s_1_best_s1 = tf.cast(tf.math.less(y_s1, y_1_s1), tf.int64)
    y_best_predicted_1 = s_best_s1 * y_s1 + s_1_best_s1 * y_1_s1

    # select = tf.cast(select, tf.float32)                                # ésta está mal?
    # select_1 = tf.cast(select_1, tf.float32)                            # ésta está mal?
    select_1 = tf.cast(s_best_s1, tf.float32)  # se corrigió (ésta es la correcta)
    select_1_1 = tf.cast(s_1_best_s1, tf.float32)  # se corrigió (ésta es la correcta)
    select_1 = tf.expand_dims(select_1, 1)
    select_1_1 = tf.expand_dims(select_1_1, 1)

    x_best_predicted_1 = (x * select_1) + (x_1 * select_1_1)

    # Second predicted digit is equal to any of the original two digits --------------------------
    s_best_s2 = tf.cast(tf.math.greater_equal(y_s2, y_1_s2), tf.int64)
    s_1_best_s2 = tf.cast(tf.math.less(y_s2, y_1_s2), tf.int64)
    y_best_predicted_2 = s_best_s2 * y_s2 + s_1_best_s2 * y_1_s2

    # select = tf.cast(select, tf.float32)                                # ésta está mal?
    # select_1 = tf.cast(select_1, tf.float32)                            # ésta está mal?
    select_1 = tf.cast(s_best_s2, tf.float32)  # se corrigió (ésta es la correcta)
    select_1_1 = tf.cast(s_1_best_s2, tf.float32)  # se corrigió (ésta es la correcta)
    select_1 = tf.expand_dims(select_1, 1)
    select_1_1 = tf.expand_dims(select_1_1, 1)

    #x_best_predicted_2 = (x * select_1) + (x_1 * select_1_1)

    # Select the best image based on y_predicted_mix_orig (y or y_1)
    select = tf.cast(tf.math.equal(y_reduced, y_predicted_mix_orig), tf.int64)
    select_1 = tf.cast(tf.math.equal(y_1_reduced, y_predicted_mix_orig), tf.int64)
    y_s = y_reduced * select
    y_s1 = y_1_reduced * select_1
    s_best_s = tf.cast(tf.math.greater_equal(y_s, y_s1), tf.int64)
    s_best_s1 = tf.cast(tf.math.less(y_s, y_s1), tf.int64)
    #y_best = s_best_s * y_s + s_best_s1 * y_s1

    # MSE ------------------------------------------------------------------------
    MSE = tf.math.reduce_mean(tf.keras.metrics.MSE(x, x_decoded_1))
    # print("MSE: ", MSE.numpy())
    MSE_1 = tf.math.reduce_mean(tf.keras.metrics.MSE(x_1, x_decoded_1))
    # print("MSE_1: ", MSE_1.numpy())
    MSE_mix = tf.math.reduce_mean(tf.keras.metrics.MSE(x_mix_filter_1, x_decoded_1))
    # print("MSE_mix: ", MSE_mix.numpy())
    MSE_mix_orig = tf.math.reduce_mean(
        tf.keras.metrics.MSE(x_mix_orig, x_decoded_1)
    )  # added
    # print("MSE_mix_orig: ", MSE_mix_orig.numpy())
    MSE_mix_best_MSE = tf.math.reduce_mean(
        tf.keras.metrics.MSE(x_best_MSE, x_decoded_1)
    )
    # print("MSE_mix_best_MSE: ", MSE_mix_best_MSE.numpy())
    MSE_mix_best_predicted = tf.math.reduce_mean(
        tf.keras.metrics.MSE(x_best_predicted_1, x_decoded_1)
    )
    # print("MSE_mix_best_predicted: ", MSE_mix_best_predicted.numpy())
    MSE_mix_orig_mix = tf.math.reduce_mean(
        tf.keras.metrics.MSE(x_mix_orig, x_mix_filter_1)
    )  # added
    # print("MSE_mix_orig_mix: ", MSE_mix_orig_mix.numpy())

    
    # Accuracy -------------------------------------------------------------------



    #-----------------------------------------------------------------------------
    m = tf.keras.metrics.Accuracy()
    m.reset_state()
    m.update_state(y_predicted, y_reduced)
    # print("Accuracy(y_predicted, y_reduced): ", m.result().numpy())
    m.reset_state()
    m.update_state(y_predicted_mix, y_reduced)
    # print("Accuracy(y_predicted_mix, y_reduced): ", m.result().numpy())
    m.reset_state()
    m.update_state(y_predicted_mix, y_1_reduced)
    # print("Accuracy(y_predicted_mix, y_1_reduced): ", m.result().numpy())
    m.reset_state()
    m.update_state(y_predicted_1_f, y_reduced)
    # print("Accuracy(y_predicted_1_f, y_reduced): ", m.result().numpy())
    m.reset_state()
    m.update_state(y_predicted_1_f, y_1_reduced)
    # print("Accuracy(y_predicted_1_f, y_1_reduced): ", m.result().numpy())
    m.reset_state()
    m.update_state(y_predicted_mix, y_best_predicted_1)
    # print("Accuracy(y_predicted_mix, y_best_predicted_1): ", m.result().numpy())
    m.reset_state()
    m.update_state(y_predicted_1_f, y_best_predicted_1)
    # print("Accuracy(y_predicted_1_f, y_best_predicted_1): ", m.result().numpy())
    m.reset_state()
    m.update_state(y_predicted_2_f, y_best_predicted_2)
    # print("Accuracy(y_predicted_2_f, y_best_predicted_2): ", m.result().numpy())
    m.reset_state()
    m.update_state(y_predicted_mix_orig, y_best_predicted_1)
    # print("Accuracy(y_predicted_mix_orig, y_best_predicted_1): ", m.result().numpy())

 
    # Global accuracy -----------------------------------------------------------------------
    
    mask = tf.ones_like(select_1_OR)
    m.reset_state()
    m.update_state(select_1_OR, mask)
    #  print("Accuracy(select_1_OR, mask): ", m.result().numpy())

    mask = tf.ones_like(select_2_OR)
    m.reset_state()
    m.update_state(select_2_OR, mask)
    #  print("Accuracy(select_2_OR, mask): ", m.result().numpy())

    mask = tf.ones_like(s_best_OR)
    m.reset_state()
    m.update_state(s_best_OR, mask)
    # print("Accuracy(s_best_OR, mask): ", m.result().numpy())

    select_reduced = tf.cast(tf.math.not_equal(y_reduced, y_1_reduced), tf.int64)
    select_12_f = tf.cast(tf.math.equal(y_predicted_1_f, y_predicted_2_f), tf.int64)
    s_reduced_12_f_AND = select_reduced * select_12_f
    s_best_AND_AND = (s_best_AND - s_reduced_12_f_AND) * select_1_OR

    m.reset_state()
    m.update_state(s_best_AND_AND, mask)

    gt1 = x
    gt2 = x_1
    gt_mix = x_mix_orig
    gen1 = x_mix_filter_1
    gen2 = x_mix_filter_2
    gen1_d = x_decoded_1
    gen2_d = x_decoded_2

    bpsnr = met.batched_psnr(gt1, gt2, gen1, gen2)  # contra el digito filtrado
    bpsnr_d = met.batched_psnr(gt1, gt2, gen1_d, gen2_d)  # contra las mascaras

    
   # acc_at_least_one , acc_both = met.accuracys(p1=   ,p2,y1,y2)
     


    return (x_best_predicted_1, y_predicted_1_f, y_predicted_2_f, bpsnr,bpsnr_d)
    """

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
