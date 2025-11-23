from project.CROP.models.crop_base_model import CropBaseModel
from project.custom_layers.sampling import Sampling

import tensorflow as tf



class CropTest(CropBaseModel):


    def filter(self, filter_1, mixed_input): 

        x_mix_filter_1 = 2 * mixed_input - filter_1
        x_mix_filter_1 = tf.convert_to_tensor(x_mix_filter_1)

        condition_encoder = self.predictor(x_mix_filter_1, verbose=0, training=False)
        print("forma de las predicciones")
        print(condition_encoder.shape)

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


    def decode(
        self,
        mixed_input,
        reconstructed_source1,
        reconstructed_source2,
        params
    ):

        mm1 = params["mm1"]
        mm2 = params["mm2"]
        rsm1 = params["rsm1"]
        rsm2 = params["rsm2"]
        # Estimación de la fuente 1
        reconstructed_source1, mask_source1, predictions_1 = (
            self.filter(
                reconstructed_source2, mixed_input
            )
        )

        # Estimación de la fuente 2
        reconstructed_source2, mask_source2, predictions_2 = (
            self.filter(
                reconstructed_source1, mixed_input
            )
        )

        
        mask_source1 = mask_source1 * mm1
        mask_source2  = mask_source2 * mm2
        reconstructed_source1 = reconstructed_source1 * rsm1
        reconstructed_source2 = reconstructed_source2 * rsm2
        
        return (
            mask_source1,
            mask_source2,
            reconstructed_source1,
            reconstructed_source2,
            predictions_1,
            predictions_2,
        )
