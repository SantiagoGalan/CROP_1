"""Crop3Test con igualdad de mezcla impuesta en cada iteracion."""

import tensorflow as tf

from project.CROP.models.crop3_test import Crop3Test


class Crop3Test2(Crop3Test):
    """Proyecta ambas fuentes sobre la restriccion de mezcla tras cada decode.

    Minimiza el cambio cuadratico de ambas estimaciones sujeto a
    alpha_mix * fuente1 + (1 - alpha_mix) * fuente2 = mezcla,
    conservando valores en [0, 1]. No hay segunda etapa ni umbral de error.
    """

    def enforce_mixture(self):
        alpha = self.model_params.get("alpha_mix", 0.5)
        if not 0 <= alpha <= 1:
            raise ValueError("alpha_mix debe estar en [0, 1]")
        source1 = tf.convert_to_tensor(self.source1_estimation)
        source2 = tf.convert_to_tensor(self.source2_estimation, dtype=source1.dtype)
        mixed = tf.convert_to_tensor(self.mixed_input, dtype=source1.dtype)
        tf.debugging.assert_greater_equal(mixed, tf.cast(0, mixed.dtype))
        tf.debugging.assert_less_equal(mixed, tf.cast(1, mixed.dtype))

        if alpha == 0:
            self.source1_estimation = tf.clip_by_value(source1, 0, 1)
            self.source2_estimation = tf.identity(mixed)
        elif alpha == 1:
            self.source1_estimation = tf.identity(mixed)
            self.source2_estimation = tf.clip_by_value(source2, 0, 1)
        else:
            # Resolver primero la fuente con menor peso evita dividir por un
            # peso pequeno al recuperar la otra a partir de la igualdad.
            if alpha <= 0.5:
                weight, first, second = alpha, source1, source2
            else:
                weight, first, second = 1 - alpha, source2, source1
            other_weight = 1 - weight
            error = weight * first + other_weight * second - mixed
            projected = first - weight * error / (weight**2 + other_weight**2)
            # Intervalo factible de la primera fuente para que la segunda
            # tambien quede en [0, 1], sin romper la igualdad.
            lower = tf.clip_by_value((mixed - other_weight) / weight, 0, 1)
            upper = tf.minimum(1., mixed / weight)
            first = tf.clip_by_value(projected, lower, upper)
            second = (mixed - weight * first) / other_weight
            if alpha <= 0.5:
                self.source1_estimation, self.source2_estimation = first, second
            else:
                self.source1_estimation, self.source2_estimation = second, first

    def decode(self):
        super().decode()
        self.enforce_mixture()
        self._debug_source = 1
        self._debug_image("Estimacion con igualdad de mezcla", self.source1_estimation)
        self._debug_source = 2
        self._debug_image("Estimacion con igualdad de mezcla", self.source2_estimation)
