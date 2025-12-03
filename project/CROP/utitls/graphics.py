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
        reconstructed_source1,
        reconstructed_source2,
        mask_source1,
        mask_source2,
        best_prediction_source1,
        model_params=None,
        metrics=None,
        title="",
        save_path=None,
        class_labels=None,
    ):
        """
        Versión mejorada:
        ✔ No muestra predicciones en métricas
        ✔ Igual estructura visual
        """

        # -----------------------------
        # 1. Construcción de la grilla
        # -----------------------------
        images = [
            mixed_input,
            source1_gt,
            source2_gt,
            reconstructed_source1,
            reconstructed_source2,
            mask_source1,
            mask_source2,
            best_prediction_source1,
        ]

        row_labels = [
            "x_mix",
            "source1_gt",
            "source2_gt",
            "x_filt_1",
            "x_filt_2",
            "x_deco_1",
            "x_deco_2",
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

        # -----------------------------
        # 2. Dibujar las imágenes
        # -----------------------------
        for row in range(num_rows):
            # Etiqueta de fila
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

        # -----------------------------
        # 3. Título general
        # -----------------------------
        fig.suptitle(title, color="darkred")

        # -----------------------------
        # 4. Texto de PARÁMETROS
        # -----------------------------
        if model_params is not None:
            param_parts = []

            for k, v in model_params.items():
                if isinstance(v, (int, float)):
                    param_parts.append(f"{k}={v:.3f}")
                else:
                    param_parts.append(f"{k}={v}")

            param_text = " | ".join(param_parts)

            fig.text(
                0.5,
                0.05,
                param_text,
                ha="center",
                color="darkblue",
                fontsize=10,
            )

        # -----------------------------
        # 5. Texto de MÉTRICAS
        # -----------------------------
        if metrics is not None:
            metric_parts = []

            for k, v in metrics.items():

                if k in ("predictions_1", "predictions_2", "best_prediction_source1"):
                    continue

                # ✔ Métrica con mean ± std
                if isinstance(v, tuple) and len(v) == 2:
                    mean, std = v
                    metric_parts.append(f"{k}={mean:.3f}")

                # ✔ Métrica simple
                elif isinstance(v, (int, float)):
                    metric_parts.append(f"{k}={v:.3f}")

                # ✔ Otros tipos
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

        # -----------------------------
        # 6. Mostrar o guardar
        # -----------------------------
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")

        plt.show()

    @classmethod
    def acc_plot(cls,acc_at_least_one_plot,acc_both_plot,plot_name=None):
    
        plt.plot(acc_at_least_one_plot, label=f"al menos uno ( {acc_at_least_one_plot[-1] } )")
        plt.plot(acc_both_plot, label=f"ambos ({acc_both_plot[-1]})")
        plt.grid()
        plt.title("Accuracy")
        plt.xlabel("iterations")
        plt.ylabel("acc")
        plt.legend()
        if plot_name:
            plt.savefig(f"{plot_name}.png")
        plt.show()




