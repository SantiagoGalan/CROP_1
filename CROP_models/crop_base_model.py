import tensorflow as tf
import numpy as np
from custom_layers.Sampling import Sampling
import inference.outcomes as out
import matplotlib.pyplot as plt
import inference.metrics as met
from abc import abstractmethod

"""
x_mix_orig → mixed_input
(the initial mixture of both sources)

x → source1_gt
(ground truth image of source 1)

x_1 → source2_gt
(ground truth image of source 2)

y → source1_cond
(conditioning vector/label for source 1)

y_1 → source2_cond
(conditioning vector/label for source 2)

x_mix_filtrado_1 → reconstructed_source1
(filtered estimate of source 1)

x_mix_filtrado_2 → reconstructed_source2
(filtered estimate of source 2)

x_decoded_1 → mask_source1
(decoder mask/activation applied to mixture for source 1)

x_decoded_2 → mask_source2
(decoder mask/activation applied to mixture for source 2)

x__x → init_placeholder
(zeros tensor used for initialization)

x_best_predicted_1 → best_prediction_source1
(final refined reconstruction of source 1 after evaluation)
"""


class CropBaseModel:
    def __init__(
        self, cvae, predictor, data, bias=None, slope=None, gamma=None, **kwargs
    ):
        self.cvae = cvae
        self.predictor = predictor
        self.use_dataset = data
        self.unmix_metrics = {}
        self.reconstruction_metrics = {}
        self.bias = 0.22 if bias is None else bias
        self.slope = 22 if slope is None else slope
        self.beta = 1
        self.alpha_1 = -2
        self.alpha_2 = -22
        self.alpha_mix = 0.5
        self.name = cvae.name
        self.gamma = 0.33 if gamma is None else gamma

    def best_filtered_var_sigmoid(self, x_mix_filter_2, mixed_input, alpha):
        # First decoded image --------------------------------------------------------------
        x_mix_filter_1 = 2 * mixed_input - x_mix_filter_2
        x_mix_filter_1 = tf.clip_by_value(
            x_mix_filter_1, clip_value_min=0, clip_value_max=1
        )
        condition_encoder = self.predictor.predict(x_mix_filter_1, verbose=0)

        condition_decoder_1 = condition_encoder

        encoded_imgs = self.cvae.encoder.predict(
            [x_mix_filter_1, condition_encoder], verbose=0
        )

        zz_log_var = encoded_imgs[1] + alpha

        z = Sampling()((encoded_imgs[0], zz_log_var))

        mask_source1 = self.cvae.decoder.predict([z, condition_decoder_1], verbose=0)
        mask_source1 = (mask_source1 - self.bias) * self.slope
        mask_source1 = tf.sigmoid(mask_source1)

        x_mix_filter_1 = 2 * mixed_input * mask_source1
        x_mix_filter_1 = tf.clip_by_value(
            x_mix_filter_1, clip_value_min=0, clip_value_max=1
        )

        return (x_mix_filter_1, mask_source1, condition_encoder)

    @abstractmethod
    def decoded_funtion(
        mixed_input,
        mask_source1,
        mask_source2,
        reconstructed_source1,
        reconstructed_source2,
        init_placeholder,
    ):
        pass

    def graphics(
        self,
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
        init_placeholder,
        best_prediction_source1,
        bpsnr=None,
        acc_at_least_one=None,
        acc_both=None,
        save_path=None,
        class_labels=None,
    ):
        # Labels en español (Fashion-MNIST)
        labels_es = [
            "remera",  # 0
            "pantalón",  # 1
            "suéter",  # 2
            "vestido",  # 3
            "campera",  # 4
            "sandalia",  # 5
            "camisa",  # 6
            "zapatilla",  # 7
            "bolso",  # 8
            "bota",  # 9
        ]

        images = [
            mixed_input,
            source1_gt,
            source2_gt,
            reconstructed_source1,
            reconstructed_source2,
            mask_source1,
            mask_source2,
            init_placeholder,
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
            "init_placeholder",
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

                # === Agregar textos sobre imágenes ===
                text = None
                color = "black"

                if row == 1:  # source1_gt → etiqueta verdadera 1
                    idx = np.argmax(source1_cond[col])
                    text = f"{labels_es[idx]}"
                    color = "blue"

                elif row == 2:  # source2_gt → etiqueta verdadera 2
                    idx = np.argmax(source2_cond[col])
                    text = f"{labels_es[idx]}"
                    color = "blue"

                elif row == 3:  # reconstructed_source1 → predicción 1
                    pred_idx = np.argmax(predictions_1[col])
                    text = f"{labels_es[pred_idx]}"
                    # Comprobar si acierta
                    y1_idx = np.argmax(source1_cond[col])
                    y2_idx = np.argmax(source2_cond[col])
                    if pred_idx in [y1_idx, y2_idx]:
                        color = "green"
                    else:
                        color = "red"

                elif row == 4:  # reconstructed_source2 → predicción 2
                    pred_idx = np.argmax(predictions_2[col])
                    text = f"{labels_es[pred_idx]}"
                    # Comprobar si acierta
                    y1_idx = np.argmax(source1_cond[col])
                    y2_idx = np.argmax(source2_cond[col])
                    if pred_idx in [y1_idx, y2_idx]:
                        color = "green"
                    else:
                        color = "red"

                if text:
                    ax.text(
                        0.5,
                        -0.1,
                        text,
                        ha="center",
                        va="top",
                        transform=ax.transAxes,
                        color=color,
                        fontsize=8,
                    )

        # ---- Título arriba con el nombre del modelo ----
        fig.suptitle(self.name, color="darkred")

        # ---- Texto de parámetros abajo ----
        param_text = f"bias={self.bias:.3f}, slope={self.slope:.3f}"
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

    def unmix(
        self,
        source1_gt,
        source2_gt,
        source1_cond,
        source2_cond,
        iterations=3,
        show_image=False,
        save_path=None,
    ):

        average_image = self.alpha_mix * source1_gt.astype(np.float32) + (
            1 - self.alpha_mix
        ) * source2_gt.astype(np.float32)
        x_mix = average_image

        # inicialmente todas las variables son el input.
        mixed_input = x_mix
        mask_source1 = x_mix
        mask_source2 = x_mix
        reconstructed_source1 = x_mix
        reconstructed_source2 = x_mix
        init_placeholder = tf.zeros_like(x_mix)

        # condition_encoder = tf.zeros_like(source1_cond)

        for j in range(iterations):
            (
                mask_source1,
                mask_source2,
                reconstructed_source1,
                reconstructed_source2,
                predictions_1,
                predictions_2,
            ) = self.decoded_funtion(
                mixed_input,
                mask_source1,
                mask_source2,
                reconstructed_source1,
                reconstructed_source2,
                init_placeholder,
            )
        (
            best_prediction_source1,
            y_predicted_s1_recon,
            y_predicted_s2_recon,
            bpsnr,
            bpsnr_d,
            acc_at_least_one,
            acc_both,
        ) = out.outcomes(
            mask_source1,
            mask_source2,
            reconstructed_source1,
            reconstructed_source2,
            mixed_input,
            source1_gt,
            source2_gt,
            source1_cond,
            source2_cond,
            self.predictor,
        )

        if show_image:
            self.graphics(
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
                init_placeholder,
                best_prediction_source1,
                bpsnr=bpsnr[0],  # mean value
                acc_at_least_one=acc_at_least_one,
                acc_both=acc_both,
                save_path=save_path,
            )
        return {
            "bpsnr": bpsnr,
            "bpsnr_d": bpsnr_d,
            "predictions_1": predictions_1,
            "predictions_2": predictions_2,
            "acc_at_least_one": acc_at_least_one,
            "acc_both": acc_both,
        }

    def get_curve(
        self,
        source1_gt,
        source2_gt,
        source1_cond,
        source2_cond,
        iterations=3,
    ):

        average_image = self.alpha_mix * source1_gt.astype(np.float32) + (
            1 - self.alpha_mix
        ) * source2_gt.astype(np.float32)
        x_mix = average_image

        # inicialmente todas las variables son el input.
        mixed_input = x_mix
        mask_source1 = x_mix
        mask_source2 = x_mix
        reconstructed_source1 = x_mix
        reconstructed_source2 = x_mix
        init_placeholder = tf.zeros_like(x_mix)

        # condition_encoder = tf.zeros_like(source1_cond)

        acc_at_least_one_plot = []
        acc_both_plot = []

        for j in range(iterations):
            (
                mask_source1,
                mask_source2,
                reconstructed_source1,
                reconstructed_source2,
                predictions_1,
                predictions_2,
            ) = self.decoded_funtion(
                mixed_input,
                mask_source1,
                mask_source2,
                reconstructed_source1,
                reconstructed_source2,
                init_placeholder,
            )

            y_predicted_s1_recon = self.predictor.predict(
                reconstructed_source1, verbose=0
            )
            y_predicted_s2_recon = self.predictor.predict(
                reconstructed_source2, verbose=0
            )

            acc_at_least_one, acc_both = met.accuracys(
                p1=y_predicted_s1_recon,
                p2=y_predicted_s2_recon,
                y1=source1_cond,
                y2=source2_cond,
            )

            acc_at_least_one_plot.append(acc_at_least_one)
            acc_both_plot.append(acc_both)

        return {
            "acc_at_least_one_plot": acc_at_least_one_plot,
            "acc_both_plot": acc_both_plot,
        }
