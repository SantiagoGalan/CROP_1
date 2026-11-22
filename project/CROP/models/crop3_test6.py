"""Crop3 con pesos de mezcla configurables y proyección por confianza."""

import tensorflow as tf

from project.CROP.models.crop3 import Crop3
from project.CROP.models.crop3_test2 import Crop3Test2
from project.CROP.models.crop3_test3 import Crop3Test3
from project.custom_layers.sampling import Sampling


class Crop3Test6(Crop3):
    """Usa alpha_mix en el filtro, la corrección gamma y la proyección.

    La confianza por píxel mide concordancia con la salida del CVAE.
    confidence_projection=False recupera la proyección uniforme.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_params.update({
            "delta": 0.353,
            "confidence_projection": True,
            "confidence_scale": 0.1,
            "confidence_floor": 0.1,
        })
        self.confidence1 = None
        self.confidence2 = None

    def enforce_mixture(self):
        if self.model_params["alpha_mix"] in (0, 1):
            Crop3Test2.enforce_mixture(self)
        else:
            Crop3Test3.enforce_mixture(self)

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
            source2 = self.source2_estimation
            source1, self.mask1, self.predictions1, self.confidence1 = self.filter(
                source2, mixed, p["alpha_2"], p["bias"], p["slope"], p["delta"], weight=a
            )
            error1 = a * source1 + b * source2 - mixed
            self.source1_estimation = tf.clip_by_value(source1 - p["gamma"] * error1, 0, 1)

            # Como en Crop3, el segundo filtro usa la primera salida previa a gamma.
            source2, self.mask2, self.predictions2, self.confidence2 = self.filter(
                source1, mixed, p["alpha_1"], p["bias"], p["slope"], p["delta"], weight=b
            )
            error2 = a * self.source1_estimation + b * source2 - mixed
            self.source2_estimation = tf.clip_by_value(source2 - p["gamma"] * error2, 0, 1)
            self.enforce_mixture()
        p["alpha_2"] *= p["beta"]
        p["alpha_1"] *= p["beta"]
