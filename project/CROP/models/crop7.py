"""Crop6 con corrección delta en píxeles con error de mezcla."""

import tensorflow as tf

from project.CROP.models.crop_base_model import CropBaseModel
from project.custom_layers.sampling import Sampling


class Crop7(CropBaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_params["gamma"] = 1.5
        self.default_params["delta"] = 0.353

    def mix(self, source1_gt, source2_gt, mix_params):
        if mix_params["alpha_mix"] != 0.5:
            raise ValueError("Crop7 requiere alpha_mix = 0.5")
        return super().mix(source1_gt, source2_gt, mix_params)

    def filter(self, other, mixed_input, alpha, bias, slope):
        estimate, mask, condition, _ = self._filter_with_decoded(
            other, mixed_input, alpha, bias, slope
        )
        return estimate, mask, condition

    def _filter_with_decoded(self, other, mixed_input, alpha, bias, slope):
        residual = tf.clip_by_value(2 * mixed_input - other, 0, 1)
        condition = self.predictor(residual, training=False)
        encoded = self.cvae.encoder([residual, condition], training=False)
        z = Sampling()((encoded[0], encoded[1] + alpha))
        decoded = self.cvae.decoder([z, condition], training=False)
        mask = tf.sigmoid((decoded - bias) * slope)
        masked_mix = mixed_input * mask
        estimate = tf.where(mixed_input > 0.5, masked_mix, 2 * masked_mix)

        mix_error = (estimate + other) / 2 - mixed_input
        estimate = estimate - self.model_params["gamma"] * mix_error

        remaining_error = (estimate + other) / 2 - mixed_input
        delta = self.model_params["delta"]
        corrected = estimate + delta * (decoded - estimate)
        estimate = tf.where(remaining_error>0.17, corrected, estimate)


        return estimate, mask, condition, decoded

    def decode(self):
        p = self.model_params
        mixed = tf.convert_to_tensor(self.mixed_input)

        self.source1_estimation, self.mask1, self.predictions1 = self.filter(
            self.source2_estimation, mixed, p["alpha_2"], p["bias"], p["slope"]
        )

        self.source2_estimation, self.mask2, self.predictions2 = self.filter(
            self.source1_estimation, mixed, p["alpha_1"], p["bias"], p["slope"]
        )
