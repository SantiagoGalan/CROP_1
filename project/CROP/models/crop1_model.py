import tensorflow as tf
from project.CROP.models.crop_base_model import CropBaseModel
from project.custom_layers.sampling import Sampling

class Crop1(CropBaseModel):


    def filter(self, filter_1, mixed_input, alpha,bias,slope):
        # def filter(self, x_mix_filter_2, mixed_input, alpha):
        # First decoded image --------------------------------------------------------------
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
        # Estimación de la fuente 1
        reconstructed_source1, mask_source1, predictions_1 = (
            self.filter(reconstructed_source2, mixed_input, alpha_2,bias,slope)
        )
        alpha_2 *= beta

        # Estimación de la fuente 2
        reconstructed_source2, mask_source2, predictions_2 = (
            self.filter(reconstructed_source1, mixed_input, alpha_1,bias,slope)
        )
        alpha_1 *= beta

        return (
            mask_source1,
            mask_source2,
            reconstructed_source1,
            reconstructed_source2,
            predictions_1,
            predictions_2,
        )
