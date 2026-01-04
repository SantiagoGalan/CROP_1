from project.CROP.models.unet_crop import CropUnetBaseModel
from project.custom_layers.sampling import Sampling
import tensorflow as tf
import numpy as np

class Crop2Unet(CropUnetBaseModel):

    def filter(self, filter_1, mixed_input, alpha, bias, slope):

        # ---------- helpers ----------
        def to_image(x):
            x = tf.convert_to_tensor(x, dtype=tf.float32)
            if len(x.shape) == 2 and x.shape[-1] == 784:
                x = tf.reshape(x, (-1, 28, 28, 1))
            elif len(x.shape) == 3:  # (B,28,28)
                x = tf.expand_dims(x, -1)
            return x

        def to_flat(x):
            x = tf.convert_to_tensor(x, dtype=tf.float32)
            if len(x.shape) == 4:
                x = tf.reshape(x, (tf.shape(x)[0], -1))
            return x

        # ---------- asegurar imagen ----------
        mixed_img  = to_image(mixed_input)
        filter_img = to_image(filter_1)

        # ---------- pre-filtrado ----------
        x_mix = 2.0 * mixed_img - filter_img
        x_mix = tf.clip_by_value(x_mix, 0.0, 1.0)

        # ---------- predictor (flatten) ----------
        x_mix_flat = to_flat(x_mix)
        condition_encoder = self.predictor(
            x_mix_flat,
            training=False
        )  # (B, 10)

        # ---------- U-Net condicional ----------
        mask_source1 = self.unet(
            [x_mix, condition_encoder],
            training=False
        )

        mask_source1 = (mask_source1 - bias) * slope
        mask_source1 = tf.sigmoid(mask_source1)

        # ---------- reconstrucción ----------
        x_rec = alpha * mixed_img * mask_source1
        x_rec = tf.clip_by_value(x_rec, 0.0, 1.0)

        return x_rec, mask_source1, condition_encoder


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