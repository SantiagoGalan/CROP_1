import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import outcomes
import PNSR
import SSIM
import importlib
import matplotlib.pyplot as plt
import numpy as np
importlib.reload(outcomes)

# Esquema 2 c (VERTICAL Y HORIZONTAL varias columnas o filas de distinto tipo) ----------------

  # def function "foto_mnist" --------------------------------------------------
def foto_mnist(x,largo):
    return (np.reshape(x,(largo,largo))*255).astype(np.uint8)

  # def function "photo_group" -------------------------------------------------
def photo_group(num_row, num_col, figsize_x, figsize_y, num_pixels,
                num_functions, num_row_group, num_col_group,
                img_group, e_img, labels_group, labels_index):

  fig, axes1 = plt.subplots((num_row * num_row_group), (num_col * num_col_group), figsize=(figsize_x, figsize_y))

  for i_group in range(num_row_group):
    for i in range(num_row):
      for j_group in range(num_col_group):
        for j in range(num_col):
            #k = np.random.choice(range(len(img_0)))                                     # ¿sacar aleatoriedad para poder comparar?
            if num_row_group < num_col_group:
              k = j_group + i_group*num_row_group
              #axes1[i][j].set_axis_off()
              axes1[i_group*num_row + i][j_group*num_col + j].imshow(foto_mnist(img_group[k,(j)+(num_col * i),:],num_pixels),\
                              interpolation='nearest', cmap='gray')
              axes1[0][j_group*num_col].xaxis.set_label_position('top')
              axes1[0][j_group*num_col].set_xlabel(e_img[j_group].numpy().decode('utf-8'), fontsize=11, color='black', loc='left')
              axes1[i][j_group*num_col + j].tick_params(labelbottom=False, labelleft=False)
              axes1[i][j_group*num_col + j].tick_params(which='both', length=0)
            else:
              k = i_group + j_group*num_col_group
              #axes1[i][j].set_axis_off()
              axes1[i_group*num_row + i][j_group*num_col + j].imshow(foto_mnist(img_group[k,(i)+(num_row * j),:],num_pixels),\
                              interpolation='nearest', cmap='gray')
              axes1[0][j_group*num_col].xaxis.set_label_position('top')
              axes1[i_group*num_row][0].set_ylabel(e_img[i_group].numpy().decode('utf-8'), fontsize=11, color='black', loc='top')
    #        axes1[0][j].set_xlabel(e_img[j].numpy().decode('utf-8'))
              axes1[i_group*num_row + i][j].tick_params(labelbottom=False, labelleft=False)
              axes1[i_group*num_row + i][j].tick_params(which='both', length=0)
            if i in labels_index:
              axes1[i][j_group*num_col + j].set_title((np.argmax(labels_group[i,0,k,:]), np.argmax(labels_group[i,1,k,:])))


def best_digit_var_sigmoid(x_mix_filtrado_2, x_mix_orig, alpha, bias, slope, predictor, encoder, decoder,vae):
    """
    Filtra y decodifica una imagen mezclada usando el encoder y decoder, aplicando un ajuste con parámetros alpha, bias y slope.
    """
    
    #mostrar_imagenes("x_mix_filtrado_2",x_mix_filtrado_2)

    
    x_mix_filtrado_1 = (2 * x_mix_orig - x_mix_filtrado_2)
    x_mix_filtrado_1 = tf.clip_by_value(x_mix_filtrado_1, clip_value_min=0, clip_value_max=1)

    #mostrar_imagenes("x_mix_filtrado_1",x_mix_filtrado_1)

    #if x_mix_filtrado_1.ndim == 2:
    #    x_mix_filtrado_1 = np.expand_dims(x_mix_filtrado_1, axis=0)

    condition_encoder = predictor.predict(x_mix_filtrado_1)
    condition_decoder_1 = condition_encoder
    #print(f"condition_decoder_1 {condition_decoder_1}")
        
    #latent_inputs = encoder.predict([x_mix_filtrado_1, condition_encoder], verbose=0)
    #x_decoded_1 = decoder.predict([latent_inputs[2], condition_decoder_1], verbose=0)
    x_decoded_1 = vae.predict([x_mix_filtrado_1, condition_encoder, condition_decoder_1], verbose=0)
   
    #mostrar_imagenes("x_decoded_1",x_decoded_1)
    
    x_decoded_1 = (x_decoded_1 - bias) * slope
    x_decoded_1 = tf.sigmoid(x_decoded_1)
    x_decoded_1 = np.squeeze(x_decoded_1)
    
    #mostrar_imagenes("x_decoded_1 despues de procesamiento",x_decoded_1)
        
    x_mix_filtrado_1 = (2 * x_mix_orig * x_decoded_1)
    
    #mostrar_imagenes("x_mix_filtrado_1",x_mix_filtrado_1)
        
    x_mix_filtrado_1 = tf.clip_by_value(x_mix_filtrado_1, clip_value_min=0, clip_value_max=1)
    
        
    #mostrar_imagenes("x_mix_filtrado_1 clip",x_mix_filtrado_1)
    
    return (x_mix_filtrado_1, x_decoded_1)



def mostrar_imagenes(titulo, imagenes, etiquetas=None, n=5):
    # Convertir a array de NumPy si es tensor
    if hasattr(imagenes, "numpy"):
        imagenes = imagenes.numpy()
    
    # Si es una sola imagen (2D o 1D), convertirla a lote de 1 imagen
    if imagenes.ndim == 2 or imagenes.ndim == 1:
        imagenes = np.expand_dims(imagenes, axis=0)
        if etiquetas is not None and not isinstance(etiquetas, (list, np.ndarray)):
            etiquetas = [etiquetas]

    cantidad = min(n, imagenes.shape[0])
    plt.figure(figsize=(cantidad * 2, 2))
    for i in range(cantidad):
        img = imagenes[i]
        # Convertir imagen plana (784,) a 28x28 si es necesario
        if img.ndim == 1:
            img = img.reshape(28, 28)
        plt.subplot(1, cantidad, i + 1)
        plt.imshow(img, cmap='gray')
        if etiquetas is not None and i < len(etiquetas):
            plt.title(str(etiquetas[i]))
        plt.axis('off')
    plt.suptitle(titulo)
    plt.show()


def inferncia_modelo(x_train, x_train_1, y_train, predictor, encoder, decoder, y_train_1,vae):
    
    img_separada_1=[]
    img_separada_2=[]
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

    Iterations = 10
    bias = 0.22 #
    slope = 22. #"entrenamiento de inferencias? para encontrar b y s "
    beta = 1.
    alpha_1 = -2
    alpha_2 = -22
    # Visualiza las imágenes originales y la mezcla inicial

    for j in range(Iterations):
        #print("="*60)
        #print(f"Numero de iteración {j}\n")
        #print("="*60)
        #print("best_digit_var_sigmoid primera vez \n")
        print(x_train_mix_filtrado_2.shape)
        print(x_train_mix_orig.shape)        
        
        x_train_mix_filtrado_1, x_train_decoded_1 = best_digit_var_sigmoid(
            x_train_mix_filtrado_2, x_train_mix_orig, alpha_2, bias, slope, predictor, encoder, decoder,vae)
        alpha_2 = alpha_2 * beta
        #print("best_digit_var_sigmoid segunda vez \n")
        x_train_mix_filtrado_2, x_train_decoded_2 = best_digit_var_sigmoid(
            x_train_mix_filtrado_1, x_train_mix_orig, alpha_1, bias, slope, predictor, encoder, decoder,vae)
        alpha_1 = alpha_1 * beta
        
        # Visualiza resultados intermedios de la inferencia
        #mostrar_imagenes(f"Iteración {j+1} - x_mix_filtrado_1", x_train_mix_filtrado_1)
        #mostrar_imagenes(f"Iteración {j+1} - x_mix_filtrado_2", x_train_mix_filtrado_2)
        #print(x_train_decoded_1.shape)
        #mostrar_imagenes(f"Iteración {j+1} - x_decoded_1", x_train_decoded_1)
        #mostrar_imagenes(f"Iteración {j+1} - x_decoded_2", x_train_decoded_2)

        # Evaluación de resultados
        _, y_train_predicted_1_f, y_train_predicted_2_f = outcomes.outcomes(
            x_train_decoded_1, x_train_decoded_2, x_train_mix_filtrado_1,
            x_train_mix_filtrado_2, x_train_mix_orig, x_train, x_train_1,
            y_train, y_train_1, predictor)


        # Begin PRINT ==================================================================
            # Parameters -----------------------------------------------------------------
        num_row = 1 #2                                                                  # Number of rows per group
        num_col = 10 #8 #10                                                                 # Number of columns per group
        num_pixels = 28
        num_functions = 9                                                               # Number of functions to be displayed (=num_row_group*num_col_group)
        num_row_group = 8                                                               # Number of group rows
        num_col_group = 1                                                               # Number of group columns
        scale_factor = 1.0                                                              # Image scale factor
        figsize_x = num_col * num_col_group * scale_factor                              # Total width of a row
        figsize_y = num_row * num_row_group * scale_factor                              # Total height of a column
            # Images ---------------------------------------------------------------------
        img_group = tf.stack([x_train_mix_orig, x_train, x_train_1, x_train_mix_filtrado_1, x_train_mix_filtrado_2, x_train_decoded_1, x_train_decoded_2, x__x])
            # Tags -----------------------------------------------------------------------
        e_img = tf.stack(["x_mix_orig", "x_train", "x_train_1", "x_filt_1", "x_filt_2", "x_deco_1", "x_deco_2", "x__x", "x_best_pred"])
            # Labels ---------------------------------------------------------------------
        labels_group = tf.stack([[y_train, y_train_1]])
        labels_index = [0]                                                              # rows with labels
            # Plot images ----------------------------------------------------------------
        photo_group(num_row, num_col, figsize_x, figsize_y, num_pixels,
                    num_functions, num_row_group, num_col_group,
                    img_group, e_img, labels_group, labels_index)
        plt.show()
        print("Fig.: En la primera fila se observan las imágenes de TRAIN superpuestas, las componentes en las dos siguientes,")
        print("      la reconstrucción final en la cuarta, la mejor imagen original basada en MSE en la quinta y en la última")
        print("      la mejor imagen según la predicción.")
        # End PRINT ====================================================================
