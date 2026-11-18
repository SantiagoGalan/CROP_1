"""Separación alternada con saturación local y proyección de mezcla ponderada."""

import tensorflow as tf

from project.CROP.models.crop3_test import Crop3Test
from project.custom_layers.sampling import Sampling


class Crop3Test3(Crop3Test):
    """La confianza es una heurística de concordancia con el CVAE, no calibrada.

    Usa siempre muestreo latente, como Crop2, y todas las iteraciones pedidas.
    confidence_projection=False recupera la proyección euclídea.
    No usa fuentes originales para refinar.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_params.update({
            "delta": 0.353,
            "gamma": 0.0,
            "saturation_object_threshold": 0.05,
            "saturation_window": 5,
            "saturation_inner_weight": 0.9,
            "confidence_projection": True,
            "confidence_scale": 0.1,
            "confidence_floor": 0.1,
        })
        self.sampling = Sampling()

    def mix(self, source1_gt, source2_gt, mix_params):
        a = mix_params["alpha_mix"]
        #asserst
        if not 0 < a < 1:
            raise ValueError("Crop3Test3 requiere 0 < alpha_mix < 1")
        window = mix_params["saturation_window"]
        if not isinstance(window, int) or window < 3 or window % 2 == 0:
            raise ValueError("saturation_window debe ser impar y >= 3")
        if not 0 <= mix_params["saturation_inner_weight"] <= 1:
            raise ValueError("saturation_inner_weight debe estar en [0, 1]")
        if mix_params["confidence_scale"] <= 0:
            raise ValueError("confidence_scale debe ser positivo")
        if not 0 < mix_params["confidence_floor"] <= 1:
            raise ValueError("confidence_floor debe estar en (0, 1]")
        if not 0 <= mix_params["delta"] <= 1:
            raise ValueError("delta debe estar en [0, 1]")
        self.iterations_run = 0
        self.change_history = []
        return super().mix(source1_gt, source2_gt, mix_params)

    @staticmethod
    def correct_local_saturation(values, object_threshold=0.05, window=5, inner_weight=0.9):
        """Combina el promedio central 3x3 y el del anillo exterior.

        inner_weight es el peso total del centro; el anillo recibe 1-peso.
        Cada promedio usa solo vecinos válidos. Se renormalizan los pesos de
        las regiones disponibles; sin peso válido se conserva el clip a 1.
        """
        if not isinstance(window, int) or window < 3 or window % 2 == 0:
            raise ValueError("window debe ser impar y >= 3")
        if not 0 <= inner_weight <= 1:
            raise ValueError("inner_weight debe estar en [0, 1]")
        x = tf.convert_to_tensor(values)
        shape = tf.shape(x)
        if x.shape.rank == 2:
            side = tf.cast(tf.sqrt(tf.cast(shape[1], tf.float32)), tf.int32)
            tf.debugging.assert_equal(side * side, shape[1])
            x = tf.reshape(x, [shape[0], side, side, 1])
        elif x.shape.rank == 3:
            x = x[..., None]
        elif x.shape.rank != 4:
            raise ValueError("Se esperaba rango 2, 3 o 4")
        valid = (x <= 1) & (x > object_threshold)
        weights = tf.cast(valid, x.dtype)
        values = tf.where(valid, x, 0)
        inner_kernel = tf.ones([3, 3, 1, 1], dtype=x.dtype)
        padding = (window - 3) // 2
        outer_kernel = tf.ones([window, window, 1, 1], dtype=x.dtype) - tf.pad(
            inner_kernel, [[padding, padding], [padding, padding], [0, 0], [0, 0]]
        )

        def region_mean(kernel):
            kernel = tf.tile(kernel, [1, 1, tf.shape(x)[-1], 1])
            total = tf.nn.depthwise_conv2d(values, kernel, [1, 1, 1, 1], "SAME")
            count = tf.nn.depthwise_conv2d(weights, kernel, [1, 1, 1, 1], "SAME")
            return tf.math.divide_no_nan(total, count), tf.cast(count > 0, x.dtype)

        inner_mean, inner_available = region_mean(inner_kernel)
        outer_mean, outer_available = region_mean(outer_kernel)
        inner_mass = inner_weight * inner_available
        outer_mass = (1 - inner_weight) * outer_available
        mass = inner_mass + outer_mass
        mean = tf.math.divide_no_nan(inner_mass * inner_mean + outer_mass * outer_mean, mass)
        corrected = tf.where((x > 1) & (mass > 0), mean, x)
        return tf.reshape(tf.clip_by_value(corrected, 0, 1), shape)

    def filter(self, other, mixed_input, alpha, bias, slope, delta):
        p = self.model_params
        weight = p["alpha_mix"] if self._debug_source == 1 else 1 - p["alpha_mix"]
        residual = tf.clip_by_value((mixed_input - (1 - weight) * other) / weight, 0, 1)
        condition = self.predictor(residual, training=False)
        encoded = self.cvae.encoder([residual, condition], training=False)
        z = self.sampling((encoded[0], encoded[1] + alpha))
        decoded = self.cvae.decoder([z, condition], training=False)
        mask = tf.sigmoid((decoded - bias) * slope)
        masked = mixed_input * mask / weight
        corrected = self.correct_local_saturation(
            masked, p["saturation_object_threshold"], p["saturation_window"],
            p["saturation_inner_weight"],
        )
        estimate = (1 - delta) * corrected + delta * decoded
        # Un error grande frente al prior permite una corrección mayor.
        confidence = tf.maximum(
            p["confidence_floor"], tf.exp(-tf.abs(estimate - decoded) / p["confidence_scale"])
        )
        self._last_confidence = confidence
        for label, image in (("Residual", residual), ("Salida CVAE", decoded),
                             ("Mascara", mask), ("Antes de saturacion local", masked),
                             ("Saturacion local corregida", corrected), ("Correccion delta", estimate)):
            self._debug_image(label, image)
        return estimate, mask, condition

    def enforce_mixture(self):
        p = self.model_params
        a, b = p["alpha_mix"], 1 - p["alpha_mix"]
        x, y = self.source1_estimation, self.source2_estimation
        c1 = self.confidence1 if p["confidence_projection"] else tf.ones_like(x)
        c2 = self.confidence2 if p["confidence_projection"] else tf.ones_like(y)
        error = a * x + b * y - self.mixed_input
        denominator = a * a / c1 + b * b / c2
        # Resolver la fuente de menor peso evita dividir por él al recuperar
        # la otra. El clip es la solución acotada del problema cuadrático.
        if a <= b:
            projected = x - (a / c1) * error / denominator
            lower = tf.maximum(0., (self.mixed_input - b) / a)
            upper = tf.minimum(1., self.mixed_input / a)
            x = tf.clip_by_value(projected, lower, upper)
            y = (self.mixed_input - a * x) / b
        else:
            projected = y - (b / c2) * error / denominator
            lower = tf.maximum(0., (self.mixed_input - a) / b)
            upper = tf.minimum(1., self.mixed_input / b)
            y = tf.clip_by_value(projected, lower, upper)
            x = (self.mixed_input - b * y) / a
        self.source1_estimation, self.source2_estimation = x, y

    def decode(self):
        p = self.model_params
        previous = (tf.convert_to_tensor(self.source1_estimation),
                    tf.convert_to_tensor(self.source2_estimation))
        a, b = p["alpha_mix"], 1 - p["alpha_mix"]
        for source, weight, other_weight, latent_key in (
            (1, a, b, "alpha_2"), (2, b, a, "alpha_1")
        ):
            self._debug_source = source
            # La segunda actualización recibe la primera ya corregida.
            other = self.source2_estimation if source == 1 else self.source1_estimation
            estimate, mask, prediction = self.filter(
                other, self.mixed_input, p[latent_key], p["bias"], p["slope"], p["delta"]
            )
            error = weight * estimate + other_weight * other - self.mixed_input
            # Para mezcla 50/50 coincide con la corrección gamma heredada.
            estimate = tf.clip_by_value(estimate - p["gamma"] * error, 0, 1)
            setattr(self, f"source{source}_estimation", estimate)
            setattr(self, f"mask{source}", mask)
            setattr(self, f"predictions{source}", prediction)
            setattr(self, f"confidence{source}", self._last_confidence)
            p[latent_key] *= p["beta"]
        self.enforce_mixture()
        # Máximo entre imágenes: un promedio global podría ocultar casos lentos.
        changes = [tf.reduce_mean(tf.abs(new - old), axis=tuple(range(1, old.shape.rank)))
                   for old, new in zip(previous, (self.source1_estimation, self.source2_estimation))]
        change = float(tf.reduce_max(tf.maximum(*changes)).numpy())
        self.change_history.append(change)
        self.iterations_run += 1
        for source in (1, 2):
            self._debug_source = source
            self._debug_image("Proyeccion de mezcla", getattr(self, f"source{source}_estimation"))
