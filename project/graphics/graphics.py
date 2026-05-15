import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class Graphics:

    @classmethod
    def complete_plot(
        cls,
        mixed_input,
        source1_gt,
        source2_gt,
        source1_labels,
        source2_labels,
        reconstructed_source1,
        reconstructed_source2,
        mask_source1,
        mask_source2,
        prediction_source1,
        prediction_source2,
        best_prediction,
        model_params=None,
        metrics=None,
        title="Separación de imagenes",
        save_path=None,
        class_labels=None,
        dataset=None,
    ):

        reconstructed_mix = 0.5 * reconstructed_source1 + 0.5 * reconstructed_source2
        reconstructed_error =  reconstructed_mix - mixed_input 
        
        # Calcular escala global para reconstructed_error
        error_vmin = float(tf.reduce_min(reconstructed_error).numpy())
        error_vmax = float(tf.reduce_max(reconstructed_error).numpy())

        images = [
            mixed_input,
            source1_gt,
            source2_gt,
            reconstructed_source1,
            reconstructed_source2,
            reconstructed_mix,
            reconstructed_error,
            mask_source1,
            mask_source2,
            best_prediction,
        ]

        row_labels = [
            "Imagen \n mezcla",
            "Imagen \n original 1",
            "Imagen \n original 2",
            "Estimacion 1",
            "Estimacion 2",
            "Mezcla \n estimada",
            "Error entre \n mezclas",
            "Mascara 1",
            "Mascara 2",
            #"Mejor estimación",
        ]

        num_rows = len(images)
        num_cols = images[0].shape[0] if len(images[0].shape) > 1 else 1
        img_size = 28

        fig_width = num_cols * 1
        fig_height = num_rows * 1
        BIG_HIGHT = 1
        SMALL_HIGHT = 0.3
        # Crear GridSpec con separaciones entre grupos de filas
        # Filas: 0(x_mix) | sep | 1(source1_gt) 2(source2_gt) | sep | 3(x_filt_1) 4(x_filt_2) | sep | 5(recon_mix) 6(recon_error) | sep | 7(mask_1) 8(mask_2) 9(x_best_pred)
        row_heights = [
            BIG_HIGHT,      # 0: x_mix
            SMALL_HIGHT,    # sep
            BIG_HIGHT, BIG_HIGHT,   # 1-2: source1_gt, source2_gt
            SMALL_HIGHT,    # sep
            BIG_HIGHT, BIG_HIGHT,   # 3-4: x_filt_1, x_filt_2
            SMALL_HIGHT,    # sep
            BIG_HIGHT, BIG_HIGHT,   # 5-6: recon_mix, recon_error
            SMALL_HIGHT,    # sep
            BIG_HIGHT, BIG_HIGHT, BIG_HIGHT # 7-9: mask_1, mask_2, x_best_pred
        ]
        
        gs = plt.GridSpec(len(row_heights), num_cols + 1, 
                          height_ratios=row_heights,
                          hspace=0.3,
                          wspace=0.1)
        
        fig = plt.figure(figsize=(fig_width + 2, fig_height + 2))
        
        # Mapeo de filas originales a filas en la grilla
        grid_row_map = [0, 2, 3, 5, 6, 8, 9, 11, 12, 13]

        for row in range(num_rows-1):
            grid_row = grid_row_map[row]
            
            # Etiqueta de la fila
            ax_label = fig.add_subplot(gs[grid_row, 0])
            ax_label.axis("off")
            ax_label.text(0.5, 0.5, row_labels[row], ha="center", va="center", fontsize=10)

            # Contenido visual
            for col in range(num_cols):
                ax = fig.add_subplot(gs[grid_row, col + 1])
                ax.axis("off")

                img = images[row][col] if num_cols > 1 else images[row]
                if len(img.shape) == 1:
                    img = tf.reshape(img, (img_size, img_size))

                # Usar escala global para recon_error
                if row_labels[row] == "Error entre \n mezclas":
                    ax.imshow(img.numpy(), cmap="gray", vmin=error_vmin, vmax=error_vmax)
                else:
                    ax.imshow(img.numpy(), cmap="gray")

                if class_labels is not None:

                    label_text = None

                    # SOURCE1_GT
                    if row_labels[row] == "Imagen \n original 1":
                        label_idx = np.argmax(source1_labels[col])
                        label_text = class_labels[label_idx]

                    # SOURCE2_GT
                    elif row_labels[row] == "Imagen \n original 2":
                        label_idx = np.argmax(source2_labels[col])
                        label_text = class_labels[label_idx]

                    # X_FILT_1 (RECONSTRUCTED SOURCE 1)
                    elif row_labels[row] == "Estimacion 1":
                        # usar la PREDICCIÓN del modelo, no la imagen reconstruida
                        if prediction_source1 is not None:
                            label_idx = np.argmax(prediction_source1[col])
                            label_text = class_labels[label_idx]

                    # X_FILT_2 (RECONSTRUCTED SOURCE 2)
                    elif row_labels[row] == "Estimacion 2":
                        if prediction_source2 is not None:
                            label_idx = np.argmax(prediction_source2[col])
                            label_text = class_labels[label_idx]

                    # Si hay label válido, imprimirlo
                    if label_text is not None:
                        ax.text(
                            0.5,
                            -0.15,
                            f"{label_text}",
                            ha="center",
                            va="center",
                            fontsize=9,
                            color="blue",
                            transform=ax.transAxes,
                        )

        

        # Construir título automáticamente si se proporciona dataset
        if dataset is not None:
            final_title = "Separación de imagenes"
        else:
            final_title = title
        
        fig.suptitle(final_title, color="darkred")

        if model_params is not None:
            param_parts = []

            for k, v in model_params.items():
                if isinstance(v, (int, float)):
                    param_parts.append(f"{k}={v:.2f}")
                else:
                    param_parts.append(f"{k}={v}")

            param_text = "Parametros: " + " | ".join(param_parts)

            fig.text(0.5,
                0.08,
                param_text,
                ha="center",
                color="darkblue",
                fontsize=10,
            )

        metric_name_map = {
            "recon_bpsnr": "BPSNR",
            "ssim": "SSIM",
            "acc_at_least_one": "Precisión (al menos uno)",
            "acc_both": "Precisión ambos objetos",
        }

        if metrics is not None:
            metric_parts = []

            for k, v in metrics.items():

                if k in ("predictions_1", "predictions_2", "best_prediction_source1", "mask_bpsnr"):
                    continue
                
                display_name = metric_name_map.get(k, k)

                if isinstance(v, tuple) and len(v) == 2:
                    mean, std = v
                    metric_parts.append(f"{display_name}={mean:.3f} ({std:.3f})")

                elif isinstance(v, (int, float)):
                    metric_parts.append(f"{display_name}={v:.3f}")

                else:
                    metric_parts.append(f"{display_name}={v}")

            metrics_text = "Resultados: " + " | ".join(metric_parts)

            fig.text(
                0.5,
                0.05,
                metrics_text,
                ha="center",
                color="black",
                fontsize=10,
            )

        
        plt.subplots_adjust(top=0.96)
        
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        
        plt.show()

    @classmethod
    def curves_plot(
        cls,
        curves_by_panel,
        model_params=None,
        title="",
        save_path=None,
    ):
        if not curves_by_panel:
            return

        # Mapeo a nombres en español
        title_map = {
            "accuracy": "Precisión vs Iteraciones",
            "accuracy_at_least_one": "Precisión vs Iteraciones",            
            "ssim": "SSIM vs Iteraciones",
            "psnr": "PSNR vs Iteraciones",
        }

        label_map = {
            "acc_both": "precisión",
            "accuracy_at_least_one": "precisión",
            "ssim": "ssim",
            "recon_psnr": "psnr",
        }

        for panel_name, series_dict in curves_by_panel.items():

            fig, ax = plt.subplots(figsize=(10, 4))

            for series_name, values in series_dict.items():
                if values is None or len(values) == 0:
                    continue

                x = np.arange(1, len(values) + 1)
                last_val = values[-1]

                metric_name = label_map.get(series_name, series_name)

                label = f"{metric_name}. último valor: {last_val:.3f}"

                # sin markers
                ax.plot(x, values, label=label)

            # Título en español
            ax.set_title(title_map.get(panel_name, panel_name))
            ax.set_xlabel("Iteraciones")

            ax.grid(True, alpha=0.3)
            ax.legend()

            # Parámetros abajo (igual que complete_plot)
            if model_params is not None:
                param_parts = []
                for k, v in model_params.items():
                    if isinstance(v, (int, float, np.floating)):
                        param_parts.append(f"{k}={float(v):.3f}")
                    else:
                        param_parts.append(f"{k}={v}")

                param_text = " | ".join(param_parts)

                fig.text(
                    0.5,
                    0.02,
                    param_text,
                    ha="center",
                    color="darkblue",
                    fontsize=10,
                )

            plt.tight_layout(rect=[0, 0.05, 1, 0.95])

            # Guardado por métrica
            if save_path:
                path = f"{save_path}_{panel_name}.png"
                plt.savefig(path, bbox_inches="tight")

            plt.show()



######################################################################################
################# Agregar gráficos que estan en "visualizaciones" ####################
######################################################################################
    @classmethod
    def reconstruction_by_condition(cls,recons):
        plt.figure(figsize=(15, 2))
        for i in range(10):
            plt.subplot(1, 10, i + 1)
            plt.imshow(recons[i].reshape(28, 28), cmap="gray")
            plt.title(f"Clase {i}")
            plt.axis("off")
        plt.suptitle("Reconstrucciones bajo distintas condiciones")
        plt.show()