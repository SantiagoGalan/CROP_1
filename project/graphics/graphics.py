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
        title="",
        save_path=None,
        class_labels=None,
    ):

        reconstructed_mix = 0.5 * reconstructed_source1 + 0.5 * reconstructed_source2
        reconstructed_error =  reconstructed_mix - mixed_input 


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
            "x_mix",
            "source1_gt",
            "source2_gt",
            "x_filt_1",
            "x_filt_2",
            "recon_mix",
            "recon_error",
            "mask_1",
            "mask_2",
            "x_best_pred",
        ]

        num_rows = len(images)
        num_cols = images[0].shape[0] if len(images[0].shape) > 1 else 1
        img_size = 28

        fig_width = num_cols * 1
        fig_height = num_rows * 1
        fig, axes = plt.subplots(num_rows, num_cols + 1, figsize=(fig_width + 2, fig_height))

        # Asegurar ejes 2D
        if num_rows == 1:
            axes = np.expand_dims(axes, 0)
        if num_cols == 1:
            axes = np.expand_dims(axes, 1)

        for row in range(num_rows):
            # Etiqueta de la fila
            ax_label = axes[row, 0]
            ax_label.axis("off")
            ax_label.text(0.5, 0.5, row_labels[row], ha="center", va="center", fontsize=10)

            # Contenido visual
            for col in range(num_cols):
                ax = axes[row, col + 1]
                ax.axis("off")

                img = images[row][col] if num_cols > 1 else images[row]
                if len(img.shape) == 1:
                    img = tf.reshape(img, (img_size, img_size))

                ax.imshow(img.numpy(), cmap="gray")

                if class_labels is not None:

                    label_text = None

                    # SOURCE1_GT
                    if row_labels[row] == "source1_gt":
                        label_idx = np.argmax(source1_labels[col])
                        label_text = class_labels[label_idx]

                    # SOURCE2_GT
                    elif row_labels[row] == "source2_gt":
                        label_idx = np.argmax(source2_labels[col])
                        label_text = class_labels[label_idx]

                    # X_FILT_1 (RECONSTRUCTED SOURCE 1)
                    elif row_labels[row] == "x_filt_1":
                        # usar la PREDICCIÓN del modelo, no la imagen reconstruida
                        if prediction_source1 is not None:
                            label_idx = np.argmax(prediction_source1[col])
                            label_text = class_labels[label_idx]

                    # X_FILT_2 (RECONSTRUCTED SOURCE 2)
                    elif row_labels[row] == "x_filt_2":
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


        fig.suptitle(title, color="darkred")

        if model_params is not None:
            param_parts = []

            for k, v in model_params.items():
                if isinstance(v, (int, float)):
                    param_parts.append(f"{k}={v:.3f}")
                else:
                    param_parts.append(f"{k}={v}")

            param_text = " | ".join(param_parts)

            fig.text(0.5,
                0.05,
                param_text,
                ha="center",
                color="darkblue",
                fontsize=10,
            )

        if metrics is not None:
            metric_parts = []

            for k, v in metrics.items():

                if k in ("predictions_1", "predictions_2", "best_prediction_source1"):
                    continue

                if isinstance(v, tuple) and len(v) == 2:
                    mean, std = v
                    metric_parts.append(f"{k}={mean:.3f}")

                elif isinstance(v, (int, float)):
                    metric_parts.append(f"{k}={v:.3f}")

                else:
                    metric_parts.append(f"{k}={v}")

            metrics_text = " | ".join(metric_parts)

            fig.text(
                0.5,
                0.00,
                metrics_text,
                ha="center",
                color="black",
                fontsize=10,
            )

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        
        plt.subplots_adjust(hspace=0.3) 
        plt.show()

    @classmethod
    def acc_plot(cls,acc_at_least_one_plot,acc_both_plot,plot_name=None):
    
        #plt.plot(acc_at_least_one_plot, label=f"al menos uno ( {acc_at_least_one_plot[-1] } )")
        plt.plot(acc_both_plot, label=f"ambos ({np.max(acc_both_plot)})")
        plt.grid()
        plt.title("Accuracy")
        plt.xlabel("iterations")
        plt.ylabel("acc")
        plt.legend()
        if plot_name:
            plt.savefig(f"{plot_name}.png")
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