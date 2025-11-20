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
        source1_cond,
        source2_cond,
        reconstructed_source1,
        reconstructed_source2,
        mask_source1,
        mask_source2,
        predictions_1,
        predictions_2,
        best_prediction_source1,
        bias,
        slope,
        title="",
        bpsnr=None,
        acc_at_least_one=None,
        acc_both=None,
        save_path=None,
        class_labels=None,
    ):

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
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))

        # Asegurar que axes siempre sea 2D
        if num_rows == 1 and num_cols == 1:
            axes = np.array([[axes]])
        elif num_rows == 1:
            axes = np.expand_dims(axes, axis=0)
        elif num_cols == 1:
            axes = np.expand_dims(axes, axis=1)

        # ---- Dibujar imágenes ----
        for row in range(num_rows):
            for col in range(num_cols):
                ax = axes[row, col]
                ax.axis("off")

                # Obtener imagen
                img = images[row][col] if num_cols > 1 else images[row]
                if len(img.shape) == 1:
                    img = tf.reshape(img, (img_size, img_size))
                img = img.numpy()
                ax.imshow(img, cmap="gray")

                # Etiquetas de fila
                if col == 0:
                    ax.set_ylabel(
                        row_labels[row],
                        labelpad=40,
                        va="center",
                        rotation=0,
                    )

        # ---- Título arriba con el nombre del modelo ----
        fig.suptitle(title, color="darkred")

        # ---- Texto de parámetros abajo ----
        param_text = f"bias={bias:.3f}, slope={slope:.3f}"
        fig.text(0.5, -0.02, param_text, ha="center", color="darkblue")
        fig.text(
            0.5,
            -0.05,
            f"bpsnr={bpsnr:.3f}, acc_one={acc_at_least_one} acc_both={acc_both}",
            ha="center",
            color="darkblue",
        )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
