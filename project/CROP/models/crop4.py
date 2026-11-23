"""Crop3 con pesos de mezcla configurables y proyección por confianza."""

import tensorflow as tf

from project.CROP.models.crop_base_model import CropBaseModel
from project.custom_layers.sampling import Sampling


class Crop4(CropBaseModel):
    """Usa alpha_mix en el filtro y la proyección.

    La confianza por píxel mide concordancia con la salida del CVAE.
    confidence_projection=False recupera la proyección uniforme.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_params.pop("gamma", None)
        self.default_params.update({
            "delta": 0.353,
            "confidence_projection": True,
            "confidence_scale": 0.1,
            "confidence_floor": 0.1,
        })
        self.confidence1 = None
        self.confidence2 = None

    def enforce_mixture(self):
        params = self.model_params
        a = params["alpha_mix"]
        b = 1 - a
        x, y = self.source1_estimation, self.source2_estimation

        if a in (0, 1):
            x = tf.convert_to_tensor(x)
            y = tf.convert_to_tensor(y, dtype=x.dtype)
            mixed = tf.convert_to_tensor(self.mixed_input, dtype=x.dtype)
            tf.debugging.assert_greater_equal(mixed, tf.cast(0, mixed.dtype))
            tf.debugging.assert_less_equal(mixed, tf.cast(1, mixed.dtype))
            if a == 0:
                self.source1_estimation = tf.clip_by_value(x, 0, 1)
                self.source2_estimation = tf.identity(mixed)
            else:
                self.source1_estimation = tf.identity(mixed)
                self.source2_estimation = tf.clip_by_value(y, 0, 1)
            return

        mixed = self.mixed_input
        if params["confidence_projection"]:
            c1, c2 = self.confidence1, self.confidence2
        else:
            c1, c2 = tf.ones_like(x), tf.ones_like(y)

        error = a * x + b * y - mixed
        denominator = a * a / c1 + b * b / c2

        # Se proyecta primero la fuente de menor peso en el intervalo factible.
        if a <= b:
            projected = x - (a / c1) * error / denominator
            lower = tf.maximum(0., (mixed - b) / a)
            upper = tf.minimum(1., mixed / a)
            x = tf.clip_by_value(projected, lower, upper)
            y = (mixed - a * x) / b
        else:
            projected = y - (b / c2) * error / denominator
            lower = tf.maximum(0., (mixed - a) / b)
            upper = tf.minimum(1., mixed / b)
            y = tf.clip_by_value(projected, lower, upper)
            x = (mixed - b * y) / a
        self.source1_estimation, self.source2_estimation = x, y

    def mix(self, source1_gt, source2_gt, mix_params):
        if not 0 <= mix_params["alpha_mix"] <= 1:
            raise ValueError("alpha_mix debe estar en [0, 1]")
        if mix_params["confidence_scale"] <= 0:
            raise ValueError("confidence_scale debe ser positivo")
        if not 0 < mix_params["confidence_floor"] <= 1:
            raise ValueError("confidence_floor debe estar en (0, 1]")
        self.confidence1 = None
        self.confidence2 = None
        return super().mix(source1_gt, source2_gt, mix_params)

    def filter(self, other, mixed_input, alpha, bias, slope, delta, *, weight):
        residual = tf.clip_by_value((mixed_input - (1 - weight) * other) / weight, 0, 1)
        condition = self.predictor(residual, training=False)
        encoded = self.cvae.encoder([residual, condition], training=False)
        z = Sampling()((encoded[0], encoded[1] + alpha))
        decoded = self.cvae.decoder([z, condition], training=False)
        mask = tf.sigmoid((decoded - bias) * slope)
        estimate = tf.clip_by_value(mixed_input * mask / weight, 0, 1)
        estimate = estimate - (estimate - decoded) * delta
        p = self.model_params
        confidence = tf.maximum(
            p["confidence_floor"],
            tf.exp(-tf.abs(estimate - decoded) / p["confidence_scale"]),
        )
        return estimate, mask, condition, confidence

    def decode(self):
        p = self.model_params
        a = p["alpha_mix"]
        b = 1 - a
        mixed = tf.convert_to_tensor(self.mixed_input)
        if a in (0, 1):
            # Solo la fuente con peso unitario está determinada por la mezcla.
            self.enforce_mixture()
            self.mask1 = tf.fill(tf.shape(mixed), tf.cast(a, mixed.dtype))
            self.mask2 = tf.fill(tf.shape(mixed), tf.cast(b, mixed.dtype))
            self.predictions1 = self.predictor(self.source1_estimation, training=False)
            self.predictions2 = self.predictor(self.source2_estimation, training=False)
            floor = tf.fill(tf.shape(mixed), tf.cast(p["confidence_floor"], mixed.dtype))
            self.confidence1, self.confidence2 = (1 - b * (1 - floor), 1 - a * (1 - floor))
        else:
            self.source1_estimation, self.mask1, self.predictions1, self.confidence1 = self.filter(
                self.source2_estimation, mixed, p["alpha_2"], p["bias"], p["slope"], p["delta"], weight=a
            )

            self.source2_estimation, self.mask2, self.predictions2, self.confidence2 = self.filter(
                self.source1_estimation, mixed, p["alpha_1"], p["bias"], p["slope"], p["delta"], weight=b
            )
            self.enforce_mixture()
        p["alpha_2"] *= p["beta"]
        p["alpha_1"] *= p["beta"]
