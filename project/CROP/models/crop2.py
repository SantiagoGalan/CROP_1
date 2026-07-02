 
from project.CROP.models.crop_base_model import CropBaseModel
from project.custom_layers.sampling import Sampling
import tensorflow as tf


class Crop2(CropBaseModel):

    def filter(self, filter_1, mixed_input, alpha,bias,slope): 
        #best_filtered_var_sigmoid

        x_mix_filter_1 = 2 * mixed_input - filter_1
        x_mix_filter_1 = tf.clip_by_value(
            x_mix_filter_1, clip_value_min=0, clip_value_max=1
        )
        condition_encoder = self.predictor(x_mix_filter_1, training=False)
    
        condition_decoder_1 = condition_encoder

        encoded_imgs = self.cvae.encoder(
            [x_mix_filter_1, condition_encoder], training=0
        )

        zz_log_var = encoded_imgs[1]  + alpha

        z = Sampling()((encoded_imgs[0], zz_log_var))

        mask_source1 = self.cvae.decoder(
            [z, condition_decoder_1], training=0
        )
        mask_source1 = (mask_source1 - bias) * slope
        mask_source1 = tf.sigmoid(mask_source1)

        x_mix_filter_1 = 2 * mixed_input * mask_source1
        x_mix_filter_1 = tf.clip_by_value(
            x_mix_filter_1, clip_value_min=0, clip_value_max=1
        )

        return (x_mix_filter_1, mask_source1, condition_encoder)



    def decode(self):
        alpha_1 = self.model_params["alpha_1"]
        alpha_2 = self.model_params["alpha_2"]
        beta = self.model_params["beta"]
        bias = self.model_params["bias"]
        slope = self.model_params["slope"]
        gamma = self.model_params["gamma"]

        reconstructed_source1, mask_source1, predictions_1 = (
            self.filter(self.source2_estimation, self.mixed_input, alpha_2,bias,slope)
        )
        
     
        self.source1_estimation = reconstructed_source1
        self.mask1 = mask_source1
        self.predictions1 = predictions_1 
        self.model_params["alpha_2"] = alpha_2 * beta

        x__x = (self.source1_estimation + self.source2_estimation) / 2

        x__x_e = x__x - self.mixed_input

        self.source1_estimation = self.source1_estimation - (x__x_e * gamma)

        self.source1_estimation = tf.clip_by_value(
            self.source1_estimation, clip_value_min=0, clip_value_max=1
        )

        reconstructed_source2, mask_source2, predictions_2 = (
            self.filter(
                reconstructed_source1, self.mixed_input, alpha_1,bias,slope
            )
        )
        
        self.source2_estimation = reconstructed_source2
        self.mask2 = mask_source2
        self.predictions2 = predictions_2 
        self.model_params["alpha_1"] = alpha_1 * beta

        x__x = (self.source1_estimation + self.source2_estimation) / 2
        x__x_e = x__x - self.mixed_input

        self.source2_estimation = self.source2_estimation - (x__x_e * gamma)

        self.source2_estimation = tf.clip_by_value(
            self.source2_estimation, clip_value_min=0, clip_value_max=1
        )