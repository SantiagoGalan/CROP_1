"""CROP con correcciones gamma y delta y proyección por distancia al CVAE."""

import tensorflow as tf

from project.CROP.models.crop_base_model import CropBaseModel
from project.custom_layers.sampling import Sampling


class Crop8(CropBaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_params["gamma"] = 1.5
        self.default_params["delta"] = 0.353

    def mix(self, source1_gt, source2_gt, mix_params):
        if mix_params["alpha_mix"] != 0.5:
            raise ValueError("Crop8 requiere alpha_mix = 0.5")
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

        mix_error = (estimate + other) / 2 - mixed_input
        estimate = estimate - self.model_params["gamma"] * mix_error

        remaining_error = (estimate + other) / 2 - mixed_input
        delta = self.model_params["delta"]
        corrected = estimate + delta * (decoded - estimate)
        estimate = tf.where(remaining_error > 0.01, corrected, estimate)

        return estimate, mask, condition, decoded

    def decode(self):
        p = self.model_params
        mixed = tf.convert_to_tensor(self.mixed_input)

        self.source1_estimation, self.mask1, self.predictions1, decoded1 = self.filter(
            self.source2_estimation, mixed, p["alpha_2"], p["bias"], p["slope"]
        )
        self.source2_estimation, self.mask2, self.predictions2, decoded2 = self.filter(
            self.source1_estimation, mixed, p["alpha_1"], p["bias"], p["slope"]
        )
        self.enforce_mixture(decoded1, decoded2)

    def enforce_mixture(self, decoded1, decoded2):
        a = self.model_params["alpha_mix"]
        x, y = self.source1_estimation, self.source2_estimation

        mixed = self.mixed_input
        distance1 = tf.abs(x - decoded1)
        distance2 = tf.abs(y - decoded2)
        total_distance = distance1 + distance2
        fraction1 = tf.math.divide_no_nan(distance1, total_distance)
        fraction1 = tf.where(total_distance > 0, fraction1, 0.5)

        error = a * (x + y) - mixed
        projected = x - fraction1 * error / a
        lower = tf.maximum(0., (mixed - a) / a)
        upper = tf.minimum(1., mixed / a)
        x = tf.clip_by_value(projected, lower, upper)
        y = (mixed - a * x) / a

        self.source1_estimation, self.source2_estimation = x, y
