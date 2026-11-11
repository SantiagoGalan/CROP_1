"""Variante de Crop3Test con correccion local de saturacion por capas."""

import tensorflow as tf

from project.CROP.models.crop3_test import Crop3Test
from project.custom_layers.sampling import Sampling


class Crop3TestLayered(Crop3Test):
    @staticmethod
    def subtract_saturation_border_mean(masked_input, object_threshold=0.05):
        """Corrige solo la frontera actual, sin iteraciones internas.

        Cada pixel saturado con vecinos validos toma el promedio local de
        sus ocho vecinos con threshold < valor <= 1. Toda la frontera se
        actualiza simultaneamente y el interior permanece intacto.
        """
        image = tf.convert_to_tensor(masked_input)
        original_shape = tf.shape(image)
        if image.shape.rank == 2:
            side = tf.cast(tf.math.sqrt(tf.cast(original_shape[1], tf.float64)), tf.int32)
            tf.debugging.assert_equal(
                side * side, original_shape[1],
                message="La imagen aplanada debe ser cuadrada",
            )
            image = tf.reshape(image, [original_shape[0], side, side, 1])
        elif image.shape.rank == 3:
            image = image[..., tf.newaxis]
        elif image.shape.rank != 4:
            raise ValueError("Se esperaba una imagen con rango 2, 3 o 4")

        valid = (image > object_threshold) & (image <= 1.0)
        # El centro saturado no es valido: solo participan sus ocho vecinos.
        weights = tf.nn.avg_pool2d(tf.cast(valid, image.dtype), 3, 1, "SAME")
        values = tf.nn.avg_pool2d(
            tf.where(valid, image, tf.zeros_like(image)), 3, 1, "SAME"
        )
        layer = (image > 1.0) & (weights > 0)
        mean = tf.math.divide_no_nan(values, weights)
        subtractive_mask = tf.where(layer, image - mean, tf.zeros_like(image))
        return tf.reshape(image - subtractive_mask, original_shape)

    def mix(self, source1_gt, source2_gt, mix_params):
        # Cada nueva mezcla empieza sin correcciones de ejecuciones anteriores.
        self._saturation_subtractions = [None, None]
        self._filter_calls = 0
        return super().mix(source1_gt, source2_gt, mix_params)

    def filter(self, filter_1, mixed_input, alpha, bias, slope, delta):
        """Aplica una frontera por fuente y por iteracion externa de unmix.

        Acumula la resta antes del clip para no perder el avance entre llamadas.
        decode llama al filtro una vez para cada fuente, siempre en ese orden.
        """
        if not hasattr(self, "_saturation_subtractions"):
            self._saturation_subtractions = [None, None]
            self._filter_calls = 0
        source = self._filter_calls % 2
        x_mix_filter_1 = tf.clip_by_value(2 * mixed_input - filter_1, 0, 1)
        condition_encoder = self.predictor(x_mix_filter_1, training=False)
        encoded_imgs = self.cvae.encoder(
            [x_mix_filter_1, condition_encoder], training=False
        )
        z = Sampling()((encoded_imgs[0], encoded_imgs[1] + alpha))
        cvae_output = self.cvae.decoder([z, condition_encoder], training=False)
        mask_source1 = tf.sigmoid((cvae_output - bias) * slope)
        masked_input = 2 * mixed_input * mask_source1

        accumulated = self._saturation_subtractions[source]
        if accumulated is None:
            accumulated = tf.zeros_like(masked_input)
        pending = tf.maximum(masked_input - accumulated, 0)
        corrected = self.subtract_saturation_border_mean(
            pending,
            object_threshold=self.model_params.get("saturation_object_threshold", 0.05),
        )
        self._saturation_subtractions[source] = accumulated + (pending - corrected)
        self._filter_calls += 1

        x_mix_filter_1 = tf.clip_by_value(corrected, 0, 1)
        x_mix_filter_1 = x_mix_filter_1 - (x_mix_filter_1 - cvae_output) * delta
        return x_mix_filter_1, mask_source1, condition_encoder
