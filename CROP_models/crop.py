import tensorflow as tf
import numpy as np
from custom_layers.Sampling import Sampling
import inference.outcomes as out
import inference.fotos as ph
import matplotlib.pyplot as plt


class crop:
    def __init__(self, cvae, predictor, data, bias=None, slope=None, **kwargs):
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

    def best_filtered_var_sigmoid(
        self, x_mix_filter_2, x_mix_orig, alpha
    ):  # cambiar el nombre de x_mix_filter
        # First decoded image --------------------------------------------------------------
        x_mix_filter_1 = 2 * x_mix_orig - x_mix_filter_2  # Masked (Cochlear) x'2
        x_mix_filter_1 = tf.clip_by_value(
            x_mix_filter_1, clip_value_min=0, clip_value_max=1
        )  # Filtered mix
        condition_encoder = self.predictor.predict(
            x_mix_filter_1, verbose=0
        )  # * j * alfa     # con ponderado incremental

        condition_decoder_1 = condition_encoder

        encoded_imgs = self.cvae.encoder.predict(
            [x_mix_filter_1, condition_encoder], verbose=0
        )

        zz_log_var = encoded_imgs[1] + alpha

        z = Sampling()((encoded_imgs[0], zz_log_var))  # (z_mean, z_log_var)

        x_decoded_1 = self.cvae.decoder.predict(
            [z, condition_decoder_1], verbose=0
        )  # ver si self.cvae.decoder.predict(self.cvae.encoder.predict()) == self.cave.predict
        x_decoded_1 = (
            x_decoded_1 - self.bias
        ) * self.slope  # son parametros entrenable?
        x_decoded_1 = tf.sigmoid(x_decoded_1)

        x_mix_filter_1 = 2 * x_mix_orig * x_decoded_1  # Masked (Cochlear)
        x_mix_filter_1 = tf.clip_by_value(
            x_mix_filter_1, clip_value_min=0, clip_value_max=1
        )

        return (x_mix_filter_1, x_decoded_1, condition_encoder)

    def graphics(
        self,
        x_mix_orig,
        x,
        x_1,
        y,
        y_1,
        x_mix_filtrado_1,
        x_mix_filtrado_2,
        x_decoded_1,
        x_decoded_2,
        x__x,
        x_best_predicted_1,
        bpsnr=None,
    ):
        """
        Mostrar grupos de imágenes en filas, con etiquetas claras y texto escalado.
        """

        # Lista de imágenes y etiquetas de fila
        images = [
            x_mix_orig,
            x,
            x_1,
            x_mix_filtrado_1,
            x_mix_filtrado_2,
            x_decoded_1,
            x_decoded_2,
            x__x,
            x_best_predicted_1,
        ]
        row_labels = [
            "x_mix",
            "x_1",
            "x_2",
            "x_filt_1",
            "x_filt_2",
            "x_deco_1",
            "x_deco_2",
            "x__x",
            "x_best_pred",
        ]

        num_rows = len(images)
        num_cols = images[0].shape[0] if len(images[0].shape) > 1 else 1
        img_size = 28

        # Figsize proporcional al número de imágenes
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

        #            if col == 0:  # solo en la primera columna
        #                ax.set_ylabel(
        #                    row_labels[row],
        #                    labelpad=40,
        #                    va="center",
        #                    rotation=0,
        #                )

        for row, label in enumerate(row_labels):
            fig.text(
                0.02,  # posición X relativa
                # 1 - (row + 0.5) / num_rows,  # posición Y relativa
                1 - (row + 0.75) / num_rows,
                label,
                va="center",
                ha="right",
                fontsize=img_size * 0.4,  # escala con la imagen
                rotation=90,
            )

        # ---- Título arriba con el nombre del modelo ----
        fig.suptitle(self.name, color="darkred")

        # ---- Texto de parámetros abajo ----
        param_text = f"bias={self.bias:.3f}, slope={self.slope:.3f}"
        if bpsnr is not None:
            param_text += f", bpsnr={bpsnr:.3f}"
        fig.text(0.5, -0.02, param_text, ha="center", color="darkblue")
        # plt.tight_layout(rect=[0.08, 0, 1, 1])  # deja más espacio a la izquierda
        plt.tight_layout()
        plt.show()

    def unmix(
        self,
        x,
        x_1,
        y,
        y_1,
        Iterations=3,
        show_graph=False,
    ):

        average_image = self.alpha_mix * x.astype(np.float32) + (
            1 - self.alpha_mix
        ) * x_1.astype(np.float32)
        x_mix = average_image

        ## Initialization
        x_mix_orig = x_mix
        x_decoded_1 = (
            x_mix  # Added in order to improve the prediction in each iteration
        )

        x_decoded_2 = (
            x_mix  # Added in order to improve the prediction in each iteration
        )
        x_mix_IN = (
            x_mix  # Added in order to improve more the prediction in each iteration
        )
        x_mix_filtrado_1 = (
            x_mix  # Added in order to improve more the prediction in each iteration
        )
        x_mix_filtrado_2 = (
            x_mix  # Added in order to improve more the prediction in each iteration
        )
        x__x = tf.zeros_like(x_mix)
        # condition_encoder = tf.zeros_like(y)

        for j in range(Iterations):

            x_mix_filtrado_1, x_decoded_1, predictions_1 = (
                self.best_filtered_var_sigmoid(
                    x_mix_filtrado_2, x_mix_orig, self.alpha_2
                )
            )
            self.alpha_2 = self.alpha_2 * self.beta

            x_mix_filtrado_2, x_decoded_2, predictions_2 = (
                self.best_filtered_var_sigmoid(
                    x_mix_filtrado_1, x_mix_orig, self.alpha_1
                )
            )
            self.alpha_1 = self.alpha_1 * self.beta

        # ---- Normalización segura de máscaras (dentro del bucle) ----
        # x_decoded_1 y x_decoded_2 son las máscaras que devuelve best_filtered_var_sigmoid (sigmoids en [0,1])

        eps = 1e-6  # evita división por cero
        mask_sum = x_decoded_1 + x_decoded_2
        mask_sum = tf.maximum(mask_sum, eps)  # shape compatible

        m1 = x_decoded_1 / mask_sum
        m2 = x_decoded_2 / mask_sum

        # Reconstrucción: mantenemos el factor 2 porque x_mix es un promedio (alpha_mix=0.5)
        x_mix_filtrado_1 = tf.clip_by_value(2.0 * x_mix_orig * m1, 0.0, 1.0)
        x_mix_filtrado_2 = tf.clip_by_value(2.0 * x_mix_orig * m2, 0.0, 1.0)
        # parece haber una mejora???

        (x_best_predicted_1, _, _, bpsnr, bpsnr_d) = out.outcomes(
            x_decoded_1,
            x_decoded_2,
            x_mix_filtrado_1,
            x_mix_filtrado_2,
            x_mix_orig,
            x,
            x_1,
            y,
            y_1,
            self.predictor,
        )

        if show_graph:
            self.graphics(
                x_mix_orig,
                x,
                x_1,
                y,
                y_1,
                x_mix_filtrado_1,
                x_mix_filtrado_2,
                x_decoded_1,
                x_decoded_2,
                x__x,
                x_best_predicted_1,
                bpsnr=bpsnr[0],  # <--- pass the value
            )

        return {
            "bpsnr": bpsnr,
            "bpsnr_d": bpsnr_d,
            "predictions_1": predictions_1,
            "predictions_2": predictions_2,
        }

    def reconstruct(self, input_image, intput_cond, output_cond=None, title=""):

        _, _, z = self.cvae.encoder.predict([input_image, intput_cond], verbose=0)
        reconstructed = self.cvae.decoder.predict(
            [z, intput_cond if output_cond is None else output_cond], verbose=0
        )
        plt.imshow(reconstructed.reshape(28, 28), cmap="gray")
        plt.title(title, fontsize=8)
        return reconstructed
