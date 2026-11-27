"""Crop4 con mezcla de pesos iguales y proyección ponderada por distancia."""

import tensorflow as tf

from project.CROP.models.crop_base_model import CropBaseModel
from project.custom_layers.sampling import Sampling


class Crop5(CropBaseModel):
    """Usa alpha_mix = 0.5 en el filtro y la proyección.

    La proyección reparte la corrección según el error absoluto por píxel
    respecto a la salida del CVAE.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_params.pop("gamma", None)
        self.default_params["delta"] = 0.353
        self.distance1 = None
        self.distance2 = None

    def enforce_mixture(self):
        a = self.model_params["alpha_mix"]
        x, y = self.source1_estimation, self.source2_estimation

        mixed = self.mixed_input
        total_distance = self.distance1 + self.distance2
        fraction1 = tf.math.divide_no_nan(self.distance1, total_distance)
        # Si las dos distancias son cero, ambas fuentes reciben el mismo ajuste.
        fraction1 = tf.where(total_distance > 0, fraction1, 0.5)

        error = a * (x + y) - mixed
        projected = x - fraction1 * error / a
        lower = tf.maximum(0., (mixed - a) / a)
        upper = tf.minimum(1., mixed / a)
        x = tf.clip_by_value(projected, lower, upper)
        y = (mixed - a * x) / a

        self.source1_estimation, self.source2_estimation = x, y

    def mix(self, source1_gt, source2_gt, mix_params):

        if mix_params["alpha_mix"] != 0.5:
            raise ValueError("Crop4 requiere alpha_mix = 0.5")
        self.distance1 = None
        self.distance2 = None
        return super().mix(source1_gt, source2_gt, mix_params)

    def filter(self, other, mixed_input, alpha, bias, slope, delta, *, weight):
        residual = tf.clip_by_value((mixed_input - weight * other) / weight, 0, 1)
        condition = self.predictor(residual, training=False)
        encoded = self.cvae.encoder([residual, condition], training=False)
        z = Sampling()((encoded[0], encoded[1] + alpha))
        decoded = self.cvae.decoder([z, condition], training=False)
        # mask = tf.sigmoid((decoded - bias) * slope)
        mask = tf.cast(decoded >= bias, decoded.dtype)
        estimate = tf.clip_by_value(mixed_input * mask / weight, 0, 1)             
        estimate = estimate - (estimate - decoded) * delta
        # p = self.model_params
        # confidence = tf.maximum(
        #     p["confidence_floor"],
        #     tf.exp(-tf.abs(estimate - decoded) / p["confidence_scale"]),
        # )
        distance = tf.abs(estimate - decoded)
        return estimate, mask, condition, distance

    def decode(self):
        p = self.model_params
        a = p["alpha_mix"]
        mixed = tf.convert_to_tensor(self.mixed_input)

        self.source1_estimation, self.mask1, self.predictions1, self.distance1 = self.filter(
            self.source2_estimation, mixed, p["alpha_2"], p["bias"], p["slope"], p["delta"], weight=a
        )

        self.source2_estimation, self.mask2, self.predictions2, self.distance2 = self.filter(
            self.source1_estimation, mixed, p["alpha_1"], p["bias"], p["slope"], p["delta"], weight=a
        )
        self.enforce_mixture()
        p["alpha_2"] *= p["beta"]
        p["alpha_1"] *= p["beta"]
