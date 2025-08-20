import tensorflow as tf
import numpy as np
from custom_layers.Sampling import Sampling
import inference.outcomes as out
import inference.fotos as ph
import matplotlib.pyplot as plt


class crop:
    def __init__(self, cvae, predictor, data, **kwargs):
        self.cvae = cvae
        self.predictor = predictor
        self.use_dataset = data
        self.unmix_metrics = {}
        self.reconstruction_metrics = {}
        self.bias = 0.22
        self.slope = 22
        self.beta = 1
        self.alpha_1 = -2
        self.alpha_2 = -22
        self.alpha_mix = 0.5
        self.name = " "

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
    ):

        # Begin PRINT ==================================================================
        # Parameters -----------------------------------------------------------------
        num_row = 1  # 2
        num_col = 10  # Number of columns per group
        num_pixels = 28
        num_functions = (
            9  # Number of functions to be displayed (=num_row_group*num_col_group)
        )
        num_row_group = 9  # Number of group rows
        num_col_group = 1  # Number of group columns
        scale_factor = 1.0  # Image scale factor
        figsize_x = num_col * num_col_group * scale_factor  # Total width of a row
        figsize_y = num_row * num_row_group * scale_factor  # Total height of a column
        img_group = tf.stack(
            [
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
        )
        # Tags -----------------------------------------------------------------------
        e_img = tf.stack(
            [
                "x_mix_orig",
                "x",
                "x_1",
                "x_filt_1",
                "x_filt_2",
                "x_deco_1",
                "x_deco_2",
                "x__x",
                "x_best_pred",
            ]
        )
        # Labels ---------------------------------------------------------------------
        labels_group = tf.stack([[y, y_1]])
        labels_index = [0]  # rows with labels
        # Plot images ----------------------------------------------------------------

        ph.photo_group(
            num_row,
            num_col,
            figsize_x,
            figsize_y,
            num_pixels,
            num_functions,
            num_row_group,
            num_col_group,
            img_group,
            e_img,
            labels_group,
            labels_index,
        )
        plt.show()

    def unmix(
        self,
        x,
        x_1,
        y,
        y_1,
        Iterations=3,
        num_col=10,
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

        if show_graph == True:
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
