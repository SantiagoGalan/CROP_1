import tensorflow as tf
import inferencias.metricas as met 
import importlib
importlib.reload(met)
# Function *******************************************************
# OUTCOMES LIMPIO----------------

  # def function "outcomes" --------------------------------------------------
def outcomes(x_decoded_1, x_decoded_2, x_mix_filtrado_1, x_mix_filtrado_2, x_mix_orig, x, x_1, y, y_1,predictor):

  # Outcomes ---------------------------------------------------------------------
  y_predicted_1         = tf.math.argmax(predictor.predict(x_decoded_1), 1)
  y_predicted_2         = tf.math.argmax(predictor.predict(x_decoded_2), 1)
  y_predicted_1_f       = tf.math.argmax(predictor.predict(x_mix_filtrado_1), 1)
  y_predicted_2_f       = tf.math.argmax(predictor.predict(x_mix_filtrado_2), 1)
  y_predicted_mix       = tf.math.argmax(predictor.predict(x_mix_filtrado_1), 1)          #### REVISAR
  y_predicted_mix_orig  = tf.math.argmax(predictor.predict(x_mix_orig), 1)
  y_predicted      = tf.math.argmax(predictor.predict(x), 1)
  y_predicted_1    = tf.math.argmax(predictor.predict(x_1), 1)
  y_reduced        = tf.math.argmax(y, 1)
  y_1_reduced      = tf.math.argmax(y_1, 1)

    # Select the best image based on MSE
  select     = tf.cast(tf.math.less(tf.keras.metrics.MSE(x, x_decoded_1), tf.keras.metrics.MSE(x_1, x_decoded_1)), tf.float32)
  select_1   = tf.cast(tf.math.greater_equal(tf.keras.metrics.MSE(x, x_decoded_1), tf.keras.metrics.MSE(x_1, x_decoded_1)), tf.float32)
  select     = tf.expand_dims(select, 1)
  select_1   = tf.expand_dims(select_1, 1)
  x_best_MSE = (x * select) + (x_1 * select_1)

    # Select the best image based on y_predicted_1_f (y or y_1)
  select_1   = tf.cast(tf.math.equal(y_reduced, y_predicted_1_f), tf.int64)           # y   = y_predicted_1_f
  select_1_1 = tf.cast(tf.math.equal(y_1_reduced, y_predicted_1_f), tf.int64)         # y_1 = y_predicted_1_f
  select_2   = tf.cast(tf.math.equal(y_reduced, y_predicted_2_f), tf.int64)           # y   = y_predicted_2_f
  select_2_1 = tf.cast(tf.math.equal(y_1_reduced, y_predicted_2_f), tf.int64)         # y_1 = y_predicted_2_f
  print("select_1:      ", select_1)
  print("select_1_1:    ", select_1_1)
  #  select_equal = tf.cast(tf.math.equal(select, select_1), tf.int64)
  y_s1   = y_reduced * select_1
  y_1_s1 = y_1_reduced * select_1_1
  y_s2   = y_reduced * select_2
  y_1_s2 = y_1_reduced * select_2_1

  select_1_AND = select_1 * select_1_1
  select_1_OR = select_1 + select_1_1 - select_1_AND
  select_2_AND = select_2 * select_2_1
  select_2_OR = select_2 + select_2_1 - select_2_AND
  s_best_AND = select_1_OR * select_2_OR
  s_best_OR  = select_1_OR + select_2_OR - s_best_AND
  select_reduced = tf.cast(tf.math.not_equal(y_reduced, y_1_reduced), tf.int64)
  select_12_f = tf.cast(tf.math.equal(y_predicted_1_f, y_predicted_2_f), tf.int64)
  s_reduced_12_f_AND = select_reduced * select_12_f
  s_best_AND_AND = s_best_AND - s_reduced_12_f_AND

    #First predicted digit is equal to any of the original two digits --------------------------
  s_best_s1         = tf.cast(tf.math.greater_equal(y_s1, y_1_s1), tf.int64)
  s_1_best_s1       = tf.cast(tf.math.less(y_s1, y_1_s1), tf.int64)
  y_best_predicted_1 = s_best_s1 * y_s1 + s_1_best_s1 * y_1_s1

  # select = tf.cast(select, tf.float32)                                # ésta está mal?
  # select_1 = tf.cast(select_1, tf.float32)                            # ésta está mal?
  select_1           = tf.cast(s_best_s1, tf.float32)                  # se corrigió (ésta es la correcta)
  select_1_1         = tf.cast(s_1_best_s1, tf.float32)                # se corrigió (ésta es la correcta)
  select_1           = tf.expand_dims(select_1, 1)
  select_1_1         = tf.expand_dims(select_1_1, 1)
  print("################################################################################################")
  print("formas de x y select_1 \n")
  print(x.shape)
  print(x_1.shape)
  print(select_1.shape)
  print(select_1_1.shape)
  
  print("################################################################################################")
  
  #x_best_predicted_1 = ( x * select_1) + (x_1 * select_1_1)
  
  print(f"forma de x y de x_1 antes de aplanar: x: {x.shape} x_1: {x_1.shape}")
  
  x_flat = tf.reshape(x,(x.shape[0],x.shape[1]*x.shape[2]))
  
  x_1_flat = tf.reshape(x_1,(x_1.shape[0],x_1.shape[1]*x_1.shape[2]))
  
  print(f"forma de x y de x_1 despues de aplanar: x_flat:  {x_flat.shape} y x_1_flat:  {x_1_flat.shape}")
  
  
  x_best_predicted_1 = ( x_flat * select_1) + (x_1_flat * select_1_1)
  
  

    #Second predicted digit is equal to any of the original two digits --------------------------
  s_best_s2         = tf.cast(tf.math.greater_equal(y_s2, y_1_s2), tf.int64)
  s_1_best_s2       = tf.cast(tf.math.less(y_s2, y_1_s2), tf.int64)
  y_best_predicted_2 = s_best_s2 * y_s2 + s_1_best_s2 * y_1_s2

  # select = tf.cast(select, tf.float32)                                # ésta está mal?
  # select_1 = tf.cast(select_1, tf.float32)                            # ésta está mal?
  select_1           = tf.cast(s_best_s2, tf.float32)                  # se corrigió (ésta es la correcta)
  select_1_1         = tf.cast(s_1_best_s2, tf.float32)                # se corrigió (ésta es la correcta)
  select_1           = tf.expand_dims(select_1, 1)
  select_1_1         = tf.expand_dims(select_1_1, 1)

  x_best_predicted_2 = (x_flat * select_1) + (x_1_flat * select_1_1)

  print("y_reduced:   ", y_reduced)
  print("y_1_reduced: ", y_1_reduced)
  print("y_s1:         ", y_s1)
  print("y_1_s1:        ", y_1_s1)
  print("s_best_s1:    ", s_best_s1)
  print("s_1_best_s1:   ", s_1_best_s1)
  print("y_best_predicted_1:      ", y_best_predicted_1)
  print("y_predicted_mix:  ", y_predicted_mix)
  print("y_predicted_1_f:    ", y_predicted_1_f)
  print("y_predicted_2_f:    ", y_predicted_2_f)

    # Select the best image based on y_predicted_mix_orig (y or y_1)
  select = tf.cast(tf.math.equal(y_reduced, y_predicted_mix_orig), tf.int64)
  select_1 = tf.cast(tf.math.equal(y_1_reduced, y_predicted_mix_orig), tf.int64)
  y_s = y_reduced * select
  y_s1 = y_1_reduced * select_1
  s_best_s = tf.cast(tf.math.greater_equal(y_s, y_s1), tf.int64)
  s_best_s1 = tf.cast(tf.math.less(y_s, y_s1), tf.int64)
  y_best = s_best_s * y_s + s_best_s1 * y_s1

  print("y_reduced:       ", y_reduced)
  print("y_1_reduced:     ", y_1_reduced)
  print("y_s:             ", y_s)
  print("y_s1:            ", y_s1)
  print("s_best_s:        ", s_best_s)
  print("s_best_s1:       ", s_best_s1)
  print("y_best:          ", y_best)
  print("y_predicted_mix:      ", y_predicted_mix)
  print("y_predicted_1_f:        ", y_predicted_1_f)
  print("y_predicted_mix_orig: ", y_predicted_mix_orig)


    # MSE ------------------------------------------------------------------------
  MSE = tf.math.reduce_mean(tf.keras.metrics.MSE(x, x_decoded_1))
  print("MSE: ", MSE.numpy())
  MSE_1 = tf.math.reduce_mean(tf.keras.metrics.MSE(x_1, x_decoded_1))
  print("MSE_1: ", MSE_1.numpy())
  MSE_mix = tf.math.reduce_mean(tf.keras.metrics.MSE(x_mix_filtrado_1, x_decoded_1))
  print("MSE_mix: ", MSE_mix.numpy())
  MSE_mix_orig = tf.math.reduce_mean(tf.keras.metrics.MSE(x_mix_orig, x_decoded_1))                 # added
  print("MSE_mix_orig: ", MSE_mix_orig.numpy())
  MSE_mix_best_MSE = tf.math.reduce_mean(tf.keras.metrics.MSE(x_best_MSE, x_decoded_1))
  print("MSE_mix_best_MSE: ", MSE_mix_best_MSE.numpy())
  
  print("="*60)
  print(f"formas {x_best_predicted_1.shape}   y    {x_decoded_1.shape}")
  print("="*60)
  MSE_mix_best_predicted = tf.math.reduce_mean(tf.keras.metrics.MSE(x_best_predicted_1, tf.reshape(x_decoded_1,(x_decoded_1.shape[0],x_decoded_1.shape[1] * x_decoded_1.shape[2] ) )))
  print("MSE_mix_best_predicted: ", MSE_mix_best_predicted.numpy())
  MSE_mix_orig_mix = tf.math.reduce_mean(tf.keras.metrics.MSE(x_mix_orig, x_mix_filtrado_1))            # added
  print("MSE_mix_orig_mix: ", MSE_mix_orig_mix.numpy())

    # Accuracy -------------------------------------------------------------------
  m = tf.keras.metrics.Accuracy()
  m.reset_state()
  m.update_state(y_predicted, y_reduced)
  print("Accuracy(y_predicted, y_reduced): ", m.result().numpy())
  m.reset_state()
  m.update_state(y_predicted_mix, y_reduced)
  print("Accuracy(y_predicted_mix, y_reduced): ", m.result().numpy())
  m.reset_state()
  m.update_state(y_predicted_mix, y_1_reduced)
  print("Accuracy(y_predicted_mix, y_1_reduced): ", m.result().numpy())
  m.reset_state()
  m.update_state(y_predicted_1_f, y_reduced)
  print("Accuracy(y_predicted_1_f, y_reduced): ", m.result().numpy())
  m.reset_state()
  m.update_state(y_predicted_1_f, y_1_reduced)
  print("Accuracy(y_predicted_1_f, y_1_reduced): ", m.result().numpy())
  m.reset_state()
  m.update_state(y_predicted_mix, y_best_predicted_1)
  print("Accuracy(y_predicted_mix, y_best_predicted_1): ", m.result().numpy())
  m.reset_state()
  m.update_state(y_predicted_1_f, y_best_predicted_1)
  print("Accuracy(y_predicted_1_f, y_best_predicted_1): ", m.result().numpy())
  m.reset_state()
  m.update_state(y_predicted_2_f, y_best_predicted_2)
  print("Accuracy(y_predicted_2_f, y_best_predicted_2): ", m.result().numpy())
  m.reset_state()
  m.update_state(y_predicted_mix_orig, y_best_predicted_1)
  print("Accuracy(y_predicted_mix_orig, y_best_predicted_1): ", m.result().numpy())

    # Global accuracy -----------------------------------------------------------------------

  L = 60

  mask = tf.ones_like(select_1_OR)
  m.reset_state()
  m.update_state(select_1_OR, mask)
  print("Accuracy(select_1_OR, mask): ", m.result().numpy())

  mask = tf.ones_like(select_2_OR)
  m.reset_state()
  m.update_state(select_2_OR, mask)
  print("Accuracy(select_2_OR, mask): ", m.result().numpy())

  mask = tf.ones_like(s_best_OR)
  m.reset_state()
  m.update_state(s_best_OR, mask)
  print("Accuracy(s_best_OR, mask): ", m.result().numpy())

  select_reduced = tf.cast(tf.math.not_equal(y_reduced, y_1_reduced), tf.int64)
  select_12_f = tf.cast(tf.math.equal(y_predicted_1_f, y_predicted_2_f), tf.int64)
  s_reduced_12_f_AND = select_reduced * select_12_f
  s_best_AND_AND = (s_best_AND - s_reduced_12_f_AND) * select_1_OR

  m.reset_state()
  m.update_state(s_best_AND_AND, mask)
  print("Accuracy(s_best_AND_AND, mask): ", m.result().numpy())


  print(y_reduced[0:L])
  print(y_1_reduced[0:L])
  print(tf.math.argmax(predictor.predict(x_mix_filtrado_1), 1)[0:L])
  print(tf.math.argmax(predictor.predict(x_mix_filtrado_2), 1)[0:L])


  ###### ACÁ VA SSIM ##############################################################


  ###### ACÁ VA PSNR ##############################################################

  gt1    = x
  gt2    = x_1
  gt_mix = x_mix_orig
  gen1   = x_mix_filtrado_1
  gen2   = x_mix_filtrado_2
  gen1_d = x_decoded_1
  gen2_d = x_decoded_2

  bpsnr_mean   = met.batched_psnr(gt1, gt2, gen1, gen2)
  bpsnr_mean_d =  met.batched_psnr(gt1, gt2, gen1_d, gen2_d)
#  bpsnr_mean_mix = psnr_grayscale(gt_mix, preds)

  print("bpsnr_mean = ", bpsnr_mean)
  print("bpsnr_mean_d = ", bpsnr_mean_d)


  return (x_best_predicted_1, y_predicted_1_f, y_predicted_2_f)
