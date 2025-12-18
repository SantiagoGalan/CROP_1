from project.CROP.models.crop_base_model import CropBaseModel
from project.custom_layers.sampling import Sampling

import tensorflow as tf



class CropTest(CropBaseModel):


    def filter(self, filter_1, mixed_input): 

        x_mix_filter_1 = 2 * mixed_input - filter_1
        x_mix_filter_1 = tf.convert_to_tensor(x_mix_filter_1)

        condition_encoder = self.predictor(x_mix_filter_1, verbose=0, training=False)
    
        condition_decoder_1 = condition_encoder

        encoded_imgs = self.cvae.encoder(
            [x_mix_filter_1, condition_encoder], verbose=0, training=0
        )

        zz_log_var = encoded_imgs[1]

        z = Sampling()((encoded_imgs[0], zz_log_var))

        mask_source1 = self.cvae.decoder(
            [z, condition_decoder_1], verbose=0, Training=False
        )

        return (x_mix_filter_1, mask_source1, condition_encoder)


    def decode(self):

        alpha_1 = self.model_params["alpha_1"]
        alpha_2 = self.model_params["alpha_2"]
        beta = self.model_params["beta"]
        bias = self.model_params["bias"]

        reconstructed_source1, mask_source1, predictions_1 = (
            self.filter(
                self.source2_estimation, self.mixed_input
            )
        )

        reconstructed_source2, mask_source2, predictions_2 = (
            self.filter(
                self.source1_estimation, self.mixed_input
            )
        )
        
        self.mask_source1 = mask_source1 * alpha_1
        self.mask_source2  = mask_source2 * alpha_2
        self.source1_estimation = reconstructed_source1 * beta
        self.source2_estimation = reconstructed_source2 * bias
        self.predictions1 = predictions_1
        self.predictions2 = predictions_2 
        
