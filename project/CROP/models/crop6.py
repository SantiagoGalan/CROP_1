"""CROP con corrección gamma y proyección ortogonal sobre la recta de mezcla."""

import tensorflow as tf

from project.CROP.models.crop_base_model import CropBaseModel
from project.custom_layers.sampling import Sampling


class Crop6(CropBaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_params["gamma"] = 1.5

    def mix(self, source1_gt, source2_gt, mix_params):
        if mix_params["alpha_mix"] != 0.5:
            raise ValueError("Crop6 requiere alpha_mix = 0.5")
        return super().mix(source1_gt, source2_gt, mix_params)

    def filter(self, other, mixed_input, alpha, bias, slope):
        residual = tf.clip_by_value(2 * mixed_input - other, 0, 1)
        condition = self.predictor(residual, training=False)
        encoded = self.cvae.encoder([residual, condition], training=False)
        z = Sampling()((encoded[0], encoded[1] + alpha))
        decoded = self.cvae.decoder([z, condition], training=False)
        mask = tf.sigmoid((decoded - bias) * slope)
        masked_mix = mixed_input * mask
        estimate = tf.where(mixed_input > 0.5, masked_mix, 2 * masked_mix)
        estimate = tf.clip_by_value(estimate, 0, 1)

        return estimate, mask, condition

    def decode(self):
        p = self.model_params
        gamma = p["gamma"]
        mixed = tf.convert_to_tensor(self.mixed_input)

        reconstructed1, self.mask1, self.predictions1 = self.filter(
            self.source2_estimation, mixed, p["alpha_2"], p["bias"], p["slope"]
        )
        error = (reconstructed1 + self.source2_estimation) / 2 - mixed
        self.source1_estimation = reconstructed1 - gamma * error

        reconstructed2, self.mask2, self.predictions2 = self.filter(
            self.source1_estimation, mixed, p["alpha_1"], p["bias"], p["slope"]
        )
        error = (self.source1_estimation + reconstructed2) / 2 - mixed
        self.source2_estimation = reconstructed2 - gamma * error
