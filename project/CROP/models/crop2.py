 
from project.CROP.models.crop_base_model import CropBaseModel
from project.custom_layers.sampling import Sampling
import tensorflow as tf


class Crop2(CropBaseModel):

    def filter(self, filter_1, mixed_input, alpha, bias, slope, *, debug_stages=None):
        #best_filtered_var_sigmoid

        x_mix_filter_1 = 2 * mixed_input - filter_1
        x_mix_filter_1 = tf.clip_by_value(
            x_mix_filter_1, clip_value_min=0, clip_value_max=1
        )
        condition_encoder = self.predictor(x_mix_filter_1, training=False)
    
        condition_decoder_1 = condition_encoder

        if debug_stages is not None:
            debug_stages.extend([
                ("filter_1\n(entrada al filtro)", filter_1),
                ("x_mix_filter_1\n(entrada al autoencoder)", x_mix_filter_1),
            ])

        encoded_imgs = self.cvae.encoder(
            [x_mix_filter_1, condition_encoder], training=0
        )

        zz_log_var = encoded_imgs[1]  + alpha

        z = Sampling()((encoded_imgs[0], zz_log_var))

        mask_source1 = self.cvae.decoder(
            [z, condition_decoder_1], training=0
        )
        if debug_stages is not None:
            debug_stages.append(("Salida del autoencoder", mask_source1))
        mask_source1 = (mask_source1 - bias) * slope
        mask_source1 = tf.sigmoid(mask_source1)

        x_mix_filter_1 = 2 * mixed_input * mask_source1
        x_mix_filter_1 = tf.clip_by_value(
            x_mix_filter_1, clip_value_min=0, clip_value_max=1
        )

        if debug_stages is not None:
            debug_stages.extend([
                ("mask_source1", mask_source1),
                ("x_mix_filter_1\n(después de la máscara)", x_mix_filter_1),
            ])
        return (x_mix_filter_1, mask_source1, condition_encoder)



    def decode(self):
        alpha_1 = self.model_params["alpha_1"]
        alpha_2 = self.model_params["alpha_2"]
        beta = self.model_params["beta"]
        bias = self.model_params["bias"]
        slope = self.model_params["slope"]
        gamma = self.model_params["gamma"]
        debug_stages = [] if self.model_params.get("debug") else None

        reconstructed_source1, mask_source1, predictions_1 = (
            self.filter(
                self.source2_estimation, self.mixed_input, alpha_2, bias, slope,
                debug_stages=debug_stages,
            )
        )
        
     
        self.source1_estimation = reconstructed_source1
        self.mask1 = mask_source1
        self.predictions1 = predictions_1 
        self.model_params["alpha_2"] = alpha_2 * beta

        mix_estimation = (self.source1_estimation + self.source2_estimation) / 2
        # 0.8        #   0.8                   +   0.8             / 2

        mixes_error = mix_estimation - self.mixed_input
        # 0.8   # 0.8   - 0
        correction = mixes_error * gamma
        self.source1_estimation = self.source1_estimation - correction
        # -0.8                            #0.8              -   (0.8*2)
        self.source1_estimation = tf.clip_by_value(
            self.source1_estimation, clip_value_min=0, clip_value_max=1
        )
        if debug_stages is not None:
            debug_stages.extend([
                ("reconstructed_source1", reconstructed_source1),
                ("self.source1_estimation\n(después de gamma)", self.source1_estimation),
            ])

        reconstructed_source2, mask_source2, predictions_2 = (
            self.filter(
                self.source1_estimation, self.mixed_input, alpha_1, bias, slope,
                debug_stages=debug_stages,
            )
        )
        
        self.source2_estimation = reconstructed_source2
        self.mask2 = mask_source2
        self.predictions2 = predictions_2 
        self.model_params["alpha_1"] = alpha_1 * beta

        mix_estimation = (self.source1_estimation + self.source2_estimation) / 2
        mixes_error = mix_estimation - self.mixed_input

        correction = mixes_error * gamma
        self.source2_estimation = self.source2_estimation - correction

        self.source2_estimation = tf.clip_by_value(
            self.source2_estimation, clip_value_min=0, clip_value_max=1
        )
        if debug_stages is not None:
            debug_stages.extend([
                ("reconstructed_source2", reconstructed_source2),
                ("self.source2_estimation\n(después de gamma)", self.source2_estimation),
            ])
            self.graphicator.crop2_debug_plot(
                debug_stages,
                title=f"{type(self).__name__} | Mezcla 1 | Iteración {self.iteration + 1}",
            )
