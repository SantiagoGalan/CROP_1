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
        # Creamos una columna extra (columna 0) para los labels
        fig, axes = plt.subplots(num_rows, num_cols + 1, figsize=(fig_width + 2, fig_height))

        # Asegurar que axes sea 2D
        if num_rows == 1:
            axes = np.expand_dims(axes, 0)
        if num_cols == 1:
            axes = np.expand_dims(axes, 1)

        # Dibujar imágenes (empiezan en columna 1)
        for row in range(num_rows):
            # Label en la columna 0
            ax_label = axes[row, 0]
            ax_label.axis("off")
            ax_label.text(
                0.5, 0.5, row_labels[row],
                ha="center", va="center",
                fontsize=10
            )

            # Imágenes desde columna 1
            for col in range(num_cols):
                ax = axes[row, col+1]
                ax.axis("off")

                img = images[row][col] if num_cols > 1 else images[row]
                if len(img.shape) == 1:
                    img = tf.reshape(img, (img_size, img_size))
                ax.imshow(img.numpy(), cmap="gray")


        # Título y textos
        fig.suptitle(title, color="darkred")

        param_text = f"bias={bias:.3f}, slope={slope:.3f}"
        fig.text(0.5, 0.02, param_text, ha="center", color="darkblue")

        fig.text(
            0.5,
            0.00,
            f"bpsnr={bpsnr:.3f}, acc_one={acc_at_least_one} acc_both={acc_both}",
            ha="center",
            color="darkblue"
        )

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




