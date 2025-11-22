from CROP_models.crop_base_model import CropBaseModel
#from  crop_base_model import CropBaseModel
import tensorflow as tf





from project.CROP.models.crop_base_model import CropBaseModel
from project.custom_layers.sampling import Sampling

# from  crop_base_model import CropBaseModel
import tensorflow as tf


class Crop2(CropBaseModel):

    def filter(self, filter_1, mixed_input, alpha,bias,slope): 
        #best_filtered_var_sigmoid

        x_mix_filter_1 = 2 * mixed_input - filter_1
        x_mix_filter_1 = tf.clip_by_value(
            x_mix_filter_1, clip_value_min=0, clip_value_max=1
        )
        condition_encoder = self.predictor(x_mix_filter_1, verbose=0, training=False)

        condition_decoder_1 = condition_encoder

        encoded_imgs = self.cvae.encoder(
            [x_mix_filter_1, condition_encoder], verbose=0, training=0
        )

        zz_log_var = encoded_imgs[1] + alpha

        z = Sampling()((encoded_imgs[0], zz_log_var))

        mask_source1 = self.cvae.decoder(
            [z, condition_decoder_1], verbose=0, Training=False
        )
        mask_source1 = (mask_source1 - bias) * slope
        mask_source1 = tf.sigmoid(mask_source1)

        x_mix_filter_1 = 2 * mixed_input * mask_source1
        x_mix_filter_1 = tf.clip_by_value(
            x_mix_filter_1, clip_value_min=0, clip_value_max=1
        )

        return (x_mix_filter_1, mask_source1, condition_encoder)



    def decode(
        self,
        mixed_input,
        mask_source1,
        mask_source2,
        reconstructed_source1,
        reconstructed_source2,
        params,

    ):
        alpha_1 = params["alpha_1"]
        alpha_2 = params["alpha_1"]
        beta = params["beta"]
        bias = params["bias"]
        slope = params["slope"]
        gamma = params["gamma"]

        reconstructed_source1, mask_source1, predictions_1 = (
            self.filter(
                reconstructed_source2, mixed_input, alpha_2,bias,slope
            )
        )

        alpha_2 = alpha_2 * beta

        x__x = (reconstructed_source1 + reconstructed_source2) / 2

        x__x_e = x__x - mixed_input

        reconstructed_source1 = reconstructed_source1 - (x__x_e * gamma)

        reconstructed_source1 = tf.clip_by_value(
            reconstructed_source1, clip_value_min=0, clip_value_max=1
        )

        reconstructed_source2, mask_source2, predictions_2 = (
            self.filter(
                reconstructed_source1, mixed_input, alpha_1,bias,slope
            )
        )

        alpha_1 = alpha_1 * beta

        x__x = (reconstructed_source1 + reconstructed_source2) / 2
        x__x_e = x__x - mixed_input

        reconstructed_source2 = reconstructed_source2 - (x__x_e * gamma)

        reconstructed_source2 = tf.clip_by_value(
            reconstructed_source2, clip_value_min=0, clip_value_max=1
        )

        return (
            mask_source1,
            mask_source2,
            reconstructed_source1,
            reconstructed_source2,
            predictions_1,
            predictions_2,
        )




class Crop2Norm(CropBaseModel):

    def decoded_funtion(
        self,
        mixed_input,
        mask_source1,
        mask_source2,
        reconstructed_source1,
        reconstructed_source2,
        init_placeholder,
    ):

        # Estimación de la fuente 1
        reconstructed_source1, mask_source1, predictions_1 = (
            self.best_filtered_var_sigmoid(
                reconstructed_source2, mixed_input, self.alpha_2
            )
        )
        self.alpha_2 *= self.beta


        # Estimación de la fuente 2
        reconstructed_source2, mask_source2, predictions_2 = (
            self.best_filtered_var_sigmoid(
                reconstructed_source1, mixed_input, self.alpha_1
            )
        )
        self.alpha_1 *= self.beta

        eps = 1e-6
        mask_sum = tf.maximum(mask_source1 + mask_source2, eps)
        m1 = mask_source1 / mask_sum
        m2 = mask_source2 / mask_sum

        reconstructed_source2 = tf.clip_by_value(2.0 * mixed_input * m2, 0.0, 1.0)
        reconstructed_source1 = tf.clip_by_value(2.0 * mixed_input * m1, 0.0, 1.0)
        
        return (
            mask_source1,
            mask_source2,
            reconstructed_source1,
            reconstructed_source2,
            predictions_1,
            predictions_2,
        )
