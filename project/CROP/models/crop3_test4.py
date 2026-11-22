"""CROP3 con corrección de saturación desde el borde hacia el interior."""

import tensorflow as tf

from project.CROP.models.crop3_test3 import Crop3Test3


class Crop3Test4(Crop3Test3):
    """Conserva Crop3Test3 y propaga la corrección local por capas.

    Usa el mismo CVAE durante toda la ejecución.
    """

    @staticmethod
    def correct_local_saturation(values, object_threshold=0.05, window=5):
        """Corrige la saturación por capas, desde el borde hacia el interior.

        Cada pasada promedia los vecinos válidos disponibles y corrige a la
        vez los píxeles saturados alcanzables. Estos pasan a ser vecinos
        válidos en la siguiente pasada. No depende del orden de recorrido.
        Si no hay más vecinos para avanzar, el resto se recorta a 1.
        """
        if not isinstance(window, int) or window < 3 or window % 2 == 0:
            raise ValueError("window debe ser impar y >= 3")
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
        pending = x > 1
        valid = (x <= 1) & (x > object_threshold)

        def can_advance(current, pending, valid, progressed):
            return tf.reduce_any(pending) & progressed

        def correct_layer(current, pending, valid, progressed):
            weights = tf.cast(valid, current.dtype)
            total = tf.nn.avg_pool2d(tf.where(valid, current, 0), window, 1, "SAME")
            count = tf.nn.avg_pool2d(weights, window, 1, "SAME")
            frontier = pending & (count > 0)
            mean = tf.math.divide_no_nan(total, count)
            # Toda la capa usa el estado previo: sus píxeles recién
            # corregidos solo participan a partir de la próxima pasada.
            corrected = tf.where(frontier, mean, current)
            return corrected, pending & ~frontier, valid | frontier, tf.reduce_any(frontier)

        corrected, _, _, _ = tf.while_loop(
            can_advance, correct_layer, (x, pending, valid, tf.constant(True))
        )
        return tf.reshape(tf.clip_by_value(corrected, 0, 1), shape)
