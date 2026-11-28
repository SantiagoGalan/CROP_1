"""CROP con correcciones gamma y delta y proyección por confianza."""

import tensorflow as tf

from project.CROP.models.crop_base_model import CropBaseModel
from project.custom_layers.sampling import Sampling


class Crop8(CropBaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_params["gamma"] = 1.5
        self.default_params["delta"] = 0.353
        self.default_params["confidence_scale"] = 0.1
        self.default_params["confidence_floor"] = 0.01

    def mix(self, source1_gt, source2_gt, mix_params):
        if mix_params["alpha_mix"] != 0.5:
            raise ValueError("Crop8 requiere alpha_mix = 0.5")
        scale = mix_params.get("confidence_scale", self.default_params["confidence_scale"])
        floor = mix_params.get("confidence_floor", self.default_params["confidence_floor"])
        if scale <= 0:
            raise ValueError("confidence_scale debe ser positivo")
        if not 0 < floor <= 1:
            raise ValueError("confidence_floor debe estar en (0, 1]")
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

        x, self.mask1, self.predictions1, decoded1 = self.filter(
            self.source2_estimation, mixed, p["alpha_2"], p["bias"], p["slope"]
        )
        y, self.mask2, self.predictions2, decoded2 = self.filter(
            x, mixed, p["alpha_1"], p["bias"], p["slope"]
        )

        scale, floor = p["confidence_scale"], p["confidence_floor"]
        c1 = tf.maximum(floor, tf.exp(-tf.abs(x - decoded1) / scale))
        c2 = tf.maximum(floor, tf.exp(-tf.abs(y - decoded2) / scale))

        a = p["alpha_mix"]
        b = 1 - a
        error = a * x + b * y - mixed
        denominator = a * a / c1 + b * b / c2
        projected = x - (a / c1) * error / denominator

        lower = tf.maximum(0., (mixed - b) / a)
        upper = tf.minimum(1., mixed / a)
        projected_x = tf.clip_by_value(projected, lower, upper)
        projected_y = (mixed - a * projected_x) / b

        needs_projection = tf.abs(error) > 0.01
        self.source1_estimation = tf.where(needs_projection, projected_x, x)
        self.source2_estimation = tf.where(needs_projection, projected_y, y)
