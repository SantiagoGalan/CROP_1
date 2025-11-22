import tensorflow as tf
import numpy as np
from project.custom_layers.sampling import Sampling
from project.CROP.utitls.graphics import Graphics
import project.inference.outcomes as out
import project.inference.metrics as met
from abc import abstractmethod, ABC

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


class CropBaseModel(ABC):
    def __init__(self, cvae, predictor, model_params=None, **kwargs):
        self.cvae = cvae
        self.predictor = predictor
        default_params = {
            "alpha_1": -2,
            "alpha_2": -22,
            "bias": 0.22,
            "slope": 22,
            "gamma": 0.33,
            "alpha_mix": 0.5,
            "beta": 1,
        }

        self.model_params = {**default_params, **(model_params or {})}

    # hacer una funcion aparte como  decoded
    @abstractmethod
    def filter(self, filter_1, mixed_input, params):
        """
        filter_1: cambiar nombre. estimacion de alguan fuente 
        mixed_innput: input orginal
        """
        pass

    @abstractmethod
    def decode(
        self,
        mixed_input,
        mask_source1,
        mask_source2,
        reconstructed_source1,
        reconstructed_source2,
        params,
    ):
        pass

    def unmix(
        self,
        source1_gt,
        source2_gt,
        source1_cond,
        source2_cond,
        iterations=3,
        show_image=False,
        save_path=None,
        params=None,
    ):
        # combinar defaults con parámetros recibidos
        params = {**self.model_params, **(params or {})}

        # usar params["alpha_mix"] como antes
        average_image = params["alpha_mix"] * source1_gt.astype(np.float32) + (
            1 - params["alpha_mix"]
        ) * source2_gt.astype(np.float32)

        mixed_input = average_image
        mask_source1 = mixed_input
        mask_source2 = mixed_input
        reconstructed_source1 = mixed_input
        reconstructed_source2 = mixed_input

        for j in range(iterations):
            (
                mask_source1,
                mask_source2,
                reconstructed_source1,
                reconstructed_source2,
                predictions_1,
                predictions_2,
            ) = self.decode(
                mixed_input,
                mask_source1,
                mask_source2,
                reconstructed_source1,
                reconstructed_source2,
                params,  # ← pasa parámetros dinámicos
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
            Graphics.complete_plot(
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
                bias=self.model_params["bias"],
                slope=self.model_params["slope"],
                title="",
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
            )

            y_predicted_s1_recon = self.predictor(
                reconstructed_source1, training=False, verbose=0
            )
            y_predicted_s2_recon = self.predictor(
                reconstructed_source2, training=False, verbose=0
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
