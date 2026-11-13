 

import numpy as np

from project.CROP.models.crop_base_model import CropBaseModel
from project.custom_layers.sampling import Sampling
import tensorflow as tf


class Crop3Test2_1(CropBaseModel):
    """Crop3Test con delta local y peso de mezcla estimado mediante barrido.

    Usa el prior 0.5 (factor 2) hasta terminar la quinta iteración;
    luego aplica el mejor peso de la iteración anterior por pareja.
    """


    def enable_debug(self):
        """Activa el registro eager de la primera imagen de cada batch."""
        self.clear_debug()
        self._debug_enabled = True

    def disable_debug(self):
        """Detiene el registro sin borrar las imagenes capturadas."""
        self._debug_enabled = False

    def clear_debug(self):
        self.debug_images = []
        self._debug_source = 0

    def _debug_image(self, stage, batch):
        if not getattr(self, "_debug_enabled", False):
            return
        # Seleccionar antes de convertir: nunca copia el batch completo a CPU.
        image = np.asarray(batch[0]).copy()
        if image.ndim == 1:
            side = int(np.sqrt(image.size))
            image = image.reshape(side, side)
        if image.ndim == 3 and image.shape[-1] == 1:
            image = image[..., 0]
        self.debug_images.append({
            "iteration": getattr(self, "iteration", 0) + 1,
            "source": getattr(self, "_debug_source", 0),
            "stage": stage,
            "image": image,
        })

    def show_debug_images(self, iteration=None, source=None, columns=4):
        """Muestra todas las capturas o filtra por iteracion/fuente (desde 1)."""
        import matplotlib.pyplot as plt

        if columns < 1:
            raise ValueError("columns debe ser mayor que cero")
        records = [r for r in getattr(self, "debug_images", [])
                   if (iteration is None or r["iteration"] == iteration)
                   and (source is None or r["source"] == source)]
        if not records:
            print("Sin imagenes de debug para mostrar. Activa enable_debug() antes de unmix().")
            return
        for start in range(0, len(records), columns * 3):
            page = records[start:start + columns * 3]
            rows = (len(page) + columns - 1) // columns
            fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.5 * rows), squeeze=False)
            for ax, record in zip(axes.flat, page):
                image = record["image"]
                finite = image[np.isfinite(image)]
                low = min(0, float(finite.min())) if finite.size else 0
                high = max(1, float(finite.max())) if finite.size else 1
                plot = ax.imshow(image, cmap="gray", vmin=low, vmax=high)
                ax.set_title(f"Iter. {record['iteration']} · Fuente {record['source']}\n"
                             f"{record['stage']}", fontsize=9)
                ax.axis("off")
                fig.colorbar(plot, ax=ax, fraction=0.046, pad=0.04)
            for ax in list(axes.flat)[len(page):]:
                ax.axis("off")
            fig.tight_layout()
            plt.show()
            plt.close(fig)

    def mix(self, source1_gt, source2_gt, mix_params):
        if getattr(self, "_debug_enabled", False):
            self.clear_debug()
        mixed = super().mix(source1_gt, source2_gt, mix_params)
        self.alpha_mix_estimation = tf.fill([tf.shape(mixed)[0]], 0.5)
        self.alpha_mix_history = []
        self.alpha_mix_valid_counts = []
        self.alpha_mix_grid = None
        self.alpha_mix_grid_errors = None
        self._alpha_saturation_masks = {}
        self._alpha_masked_inputs = {}
        return mixed

    @staticmethod
    def subtract_saturation_border_mean(masked_input, object_threshold=0.05):
        """Lleva la zona > 1 al promedio de su borde exterior no saturado.

        Usa los ocho vecinos inmediatos y un promedio por imagen y canal.
        Acepta imagenes cuadradas aplanadas [N, H*W], [N, H, W]
        o imagenes [N, H, W, C], y conserva la forma de entrada.
        Si no hay vecinos del objeto, deja la saturacion para el clip habitual.
        """
        masked_input = tf.convert_to_tensor(masked_input)
        original_shape = tf.shape(masked_input)
        if masked_input.shape.rank == 2:
            side = tf.cast(tf.math.sqrt(tf.cast(original_shape[1], tf.float64)), tf.int32)
            tf.debugging.assert_equal(
                side * side, original_shape[1],
                message="La imagen aplanada debe ser cuadrada",
            )
            masked_input = tf.reshape(masked_input, [original_shape[0], side, side, 1])
        elif masked_input.shape.rank == 3:
            masked_input = masked_input[..., tf.newaxis]
        elif masked_input.shape.rank != 4:
            raise ValueError("Se esperaba una imagen con rango 2, 3 o 4")

        saturated = masked_input > 1.0
        dilated = tf.nn.max_pool2d(
            tf.cast(saturated, masked_input.dtype),
            ksize=3, strides=1, padding="SAME",
        ) > 0
        border = dilated & ~saturated & (masked_input > object_threshold)
        border_values = tf.where(border, masked_input, tf.zeros_like(masked_input))
        count = tf.reduce_sum(
            tf.cast(border, masked_input.dtype), axis=[1, 2], keepdims=True
        )
        border_mean = tf.math.divide_no_nan(
            tf.reduce_sum(border_values, axis=[1, 2], keepdims=True), count
        )
        subtractive_mask = tf.where(
            saturated & (count > 0),
            masked_input - border_mean,
            tf.zeros_like(masked_input),
        )
        return tf.reshape(masked_input - subtractive_mask, original_shape)

    def mixture_weight(self, reference):
        """Expande el peso por imagen y evita divisiones por cero."""
        reference = tf.convert_to_tensor(reference)
        weight = tf.cast(self.alpha_mix_estimation, reference.dtype)
        weight = tf.clip_by_value(weight, 1e-4, 1 - 1e-4)
        shape = tf.concat([tf.shape(weight), tf.ones([tf.rank(reference) - 1], tf.int32)], 0)
        return tf.reshape(weight, shape)

    def filter(self, filter_1, mixed_input, alpha, bias, slope, delta, *, source_weight):

        self._debug_image("Mezcla", mixed_input)
        self._debug_image("Estimacion de la otra fuente", filter_1)
        x_mix_filter_1 = (mixed_input - (1 - source_weight) * filter_1) / source_weight
        self._debug_image("(mezcla - peso otra * otra fuente) / peso fuente", x_mix_filter_1)

        x_mix_filter_1 = tf.clip_by_value(x_mix_filter_1, 0, 1)
        self._debug_image("Entrada encoder recortada", x_mix_filter_1)
      
        condition_encoder = self.predictor(x_mix_filter_1,  training=False)
        encoded_imgs = self.cvae.encoder(
            [x_mix_filter_1, condition_encoder], training=False
        )
        
        zz_log_var = encoded_imgs[1] + alpha

        z = Sampling()((encoded_imgs[0], zz_log_var))

        cvae_output = self.cvae.decoder(
            [z, condition_encoder],  training=False
        )
        self._debug_image("Salida decoder", cvae_output)
    
        mask_source1 = (cvae_output - bias) * slope
        self._debug_image("Mascara antes de sigmoid", mask_source1)
     
        mask_source1 = tf.sigmoid(mask_source1)
        self._debug_image("Mascara sigmoid", mask_source1)
    
        x_mix_filter_1 = mixed_input * mask_source1 / source_weight
        self._debug_image("Imagen enmascarada antes de corregir", x_mix_filter_1)
        saturated = x_mix_filter_1 > 1.0
        # Captura directa de mezcla * máscara / peso, antes de correcciones.
        self._alpha_masked_inputs[self._debug_source] = tf.clip_by_value(x_mix_filter_1, 0, 1)
        self._alpha_saturation_masks[self._debug_source] = saturated
        if getattr(self, "_debug_enabled", False):
            self._debug_image("Zona saturada", tf.cast(x_mix_filter_1 > 1, x_mix_filter_1.dtype))
            before_correction = x_mix_filter_1[0:1]

        x_mix_filter_1 = self.subtract_saturation_border_mean(
            x_mix_filter_1,
            object_threshold=self.model_params.get("saturation_object_threshold", 0.05),
        )

        self._debug_image("Imagen despues de corregir saturacion", x_mix_filter_1)
        if getattr(self, "_debug_enabled", False):
            self._debug_image("Mascara sustractiva", before_correction - x_mix_filter_1[0:1])
        x_mix_filter_1 = tf.clip_by_value(x_mix_filter_1, 0, 1) 
        self._debug_image("Imagen corregida recortada", x_mix_filter_1)

        # La región se fija ANTES de corregir saturación y recortar.
        x_mix_filter_1 = tf.where(
            saturated,
            x_mix_filter_1 + delta * (cvae_output - x_mix_filter_1),
            x_mix_filter_1,
        )
        self._debug_image("Correccion delta", x_mix_filter_1)

        return x_mix_filter_1, mask_source1, condition_encoder



    def decode(self):
        alpha_1 = self.model_params["alpha_1"]
        alpha_2 = self.model_params["alpha_2"]
        beta = self.model_params["beta"]
        bias = self.model_params["bias"]
        slope = self.model_params["slope"]
        gamma = self.model_params["gamma"]
        delta = self.model_params["delta"]
        weight1 = self.mixture_weight(self.mixed_input)
        weight2 = 1 - weight1
        self._debug_source = 1
        reconstructed_source1, mask_source1, predictions_1 = (
            self.filter(self.source2_estimation, self.mixed_input, alpha_2, bias, slope, delta,
                        source_weight=weight1)
        )

        self.source1_estimation = reconstructed_source1
        self.mask1 = mask_source1
        self.predictions1 = predictions_1
        self.model_params["alpha_2"] = alpha_2 * beta

        x__x = weight1 * self.source1_estimation + weight2 * self.source2_estimation
        x__x_e = x__x - self.mixed_input
        
        self.source1_estimation = self.source1_estimation - (x__x_e * gamma)
        self._debug_image("Correccion gamma", self.source1_estimation)

        self.source1_estimation = tf.clip_by_value(self.source1_estimation, 0, 1)
        self._debug_image("Estimacion final", self.source1_estimation)
        self._debug_source = 2
        reconstructed_source2, mask_source2, predictions_2 = (
            self.filter(reconstructed_source1, self.mixed_input, alpha_1, bias, slope, delta,
                        source_weight=weight2)
        )

        self.source2_estimation = reconstructed_source2
        self.mask2 = mask_source2
        self.predictions2 = predictions_2
        self.model_params["alpha_1"] = alpha_1 * beta



        x__x = weight1 * self.source1_estimation + weight2 * self.source2_estimation
        x__x_e = x__x - self.mixed_input

        self.source2_estimation = self.source2_estimation - (x__x_e * gamma)
        self._debug_image("Correccion gamma", self.source2_estimation)


        self.source2_estimation = tf.clip_by_value(self.source2_estimation, 0, 1)
        self._debug_image("Estimacion final", self.source2_estimation)

        self.estimate_alpha_mix()

    def estimate_alpha_mix(self):
        """Barre a en [0, 1] para ajustar m = a*s1 + (1-a)*s2.

        Evalúa el error cuadrático en regiones exclusivas no saturadas.
        Al usar imágenes m*máscara/peso, el ajuste puede favorecer el peso
        previo por construcción; no identifica por sí solo el peso real.

        Usa las mezclas enmascaradas y recortadas a [0, 1], capturadas antes
        de corregir saturación, delta y gamma. No usa las estimaciones finales.
        Empieza al final de la quinta iteración y se aplica en la siguiente.
        Si faltan píxeles o contraste, conserva el valor previo (inicial: 0.5).
        """
        if self.iteration < 4:
            self.alpha_mix_estimation = tf.fill(tf.shape(self.alpha_mix_estimation), 0.5)
            self.alpha_mix_history.append(tf.identity(self.alpha_mix_estimation))
            self.alpha_mix_valid_counts.append(tf.zeros(tf.shape(self.alpha_mix_estimation), tf.int32))
            return self.alpha_mix_estimation
        first = tf.convert_to_tensor(self._alpha_masked_inputs[1])
        second = tf.cast(self._alpha_masked_inputs[2], first.dtype)
        saturated1 = self._alpha_saturation_masks[1]
        saturated2 = self._alpha_saturation_masks[2]
        mixed = tf.cast(self.mixed_input, first.dtype)
        foreground = self.model_params.get("alpha_foreground_threshold", 0.2)
        background = self.model_params.get("alpha_background_threshold", 0.05)
        if not 0 <= background < foreground <= 1:
            raise ValueError("Se requiere 0 <= background < foreground <= 1")
        exclusive1 = (first > foreground) & (second <= background)
        exclusive2 = (second > foreground) & (first <= background)
        valid = (exclusive1 | exclusive2) & ~saturated1 & ~saturated2
        valid &= tf.math.is_finite(first) & tf.math.is_finite(second)
        valid &= tf.math.is_finite(mixed)
        # El error para cada candidato es sum((a*difference-target)^2).
        # Reducir primero evita crear un tensor [batch, píxeles, candidatos].
        axes = tf.range(1, tf.rank(first))
        difference = tf.where(valid, first - second, tf.zeros_like(first))
        target = tf.where(valid, mixed - second, tf.zeros_like(first))
        numerator = tf.reduce_sum(difference * target, axis=axes)
        denominator = tf.reduce_sum(tf.square(difference), axis=axes)
        counts = tf.reduce_sum(tf.cast(valid, tf.int32), axis=axes)
        minimum = self.model_params.get("alpha_min_pixels", 5)
        if minimum < 1:
            raise ValueError("alpha_min_pixels debe ser >= 1")
        identifiable = (counts >= minimum) & (denominator > 1e-8)
        points = self.model_params.get("alpha_sweep_points", 100)
        if not isinstance(points, int) or isinstance(points, bool) or points < 2:
            raise ValueError("alpha_sweep_points debe ser un entero >= 2")
        grid = tf.linspace(tf.cast(0, first.dtype), tf.cast(1, first.dtype), points)
        target_energy = tf.reduce_sum(tf.square(target), axis=axes)
        squared_errors = (
            denominator[:, None] * tf.square(grid)[None, :]
            - 2 * numerator[:, None] * grid[None, :]
            + target_energy[:, None]
        )
        errors = tf.math.divide_no_nan(
            tf.maximum(squared_errors, 0), tf.cast(counts[:, None], first.dtype)
        )
        self.alpha_mix_grid = grid
        # Sin datos suficientes el error no es interpretable como ajuste.
        self.alpha_mix_grid_errors = tf.where(
            identifiable[:, None], errors, tf.cast(float("nan"), first.dtype)
        )
        estimate = tf.gather(grid, tf.argmin(errors, axis=1))
        self.alpha_mix_estimation = tf.where(
            identifiable, estimate, tf.cast(self.alpha_mix_estimation, first.dtype)
        )
        self.alpha_mix_history.append(tf.identity(self.alpha_mix_estimation))
        self.alpha_mix_valid_counts.append(tf.identity(counts))
        return self.alpha_mix_estimation
