import numpy as np
import matplotlib.pyplot as plt
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