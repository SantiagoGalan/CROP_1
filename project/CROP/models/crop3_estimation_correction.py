 
from project.CROP.models.crop_base_model import CropBaseModel
from project.custom_layers.sampling import Sampling
import tensorflow as tf


class Crop3EstimationCorrection(CropBaseModel):

    def filter(self, filter_1 ,mixed_input, alpha, bias, slope,delta):

        x_mix_filter_1 =  2 * mixed_input - filter_1

        x_mix_filter_1 = tf.clip_by_value(x_mix_filter_1, 0, 1)
      
        condition_encoder = self.predictor(x_mix_filter_1,  training=False)
        encoded_imgs = self.cvae.encoder(
            [x_mix_filter_1, condition_encoder], training=False
        )
        
        zz_log_var = encoded_imgs[1] + alpha

        z = Sampling()((encoded_imgs[0], zz_log_var))

        cvae_output = self.cvae.decoder(
            [z, condition_encoder],  training=False
        )
    
        mask_source1 = (cvae_output - bias) * slope
     
        mask_source1 = tf.sigmoid(mask_source1)
    
        x_mix_filter_1 = 2 * mixed_input * mask_source1

        x_mix_filter_1 = tf.clip_by_value(x_mix_filter_1, 0, 1) 

        x_mix_filter_1 = x_mix_filter_1 -(x_mix_filter_1-cvae_output)*delta

        return x_mix_filter_1, mask_source1, condition_encoder



    def decode(self):
        alpha_1 = self.model_params["alpha_1"]
        alpha_2 = self.model_params["alpha_2"]
        beta = self.model_params["beta"]
        bias = self.model_params["bias"]
        slope = self.model_params["slope"]
        gamma = self.model_params["gamma"]
        delta = self.model_params["delta"]

        reconstructed_source1, mask_source1, predictions_1 = (
            self.filter(self.source2_estimation,self.mixed_input, alpha_2, bias, slope,delta)
        )

        self.source1_estimation = reconstructed_source1
        self.mask1 = mask_source1
        self.predictions1 = predictions_1
        self.model_params["alpha_2"] = alpha_2 * beta

        x__x = (self.source1_estimation + self.source2_estimation) / 2
        x__x_e = x__x - self.mixed_input
        
        self.source1_estimation = self.source1_estimation - (x__x_e * gamma)

        self.source1_estimation = tf.clip_by_value(self.source1_estimation, 0, 1)
        reconstructed_source2, mask_source2, predictions_2 = (
            self.filter(reconstructed_source1 ,self.mixed_input, alpha_1, bias, slope,delta)
        )

        self.source2_estimation = reconstructed_source2
        self.mask2 = mask_source2
        self.predictions2 = predictions_2
        self.model_params["alpha_1"] = alpha_1 * beta

        x__x = (self.source1_estimation + self.source2_estimation) / 2
        x__x_e = x__x - self.mixed_input

        self.source2_estimation = self.source2_estimation - (x__x_e * gamma)


        self.source2_estimation = tf.clip_by_value(self.source2_estimation, 0, 1)
