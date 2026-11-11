 
from abc import abstractmethod

import numpy as np

from project.CROP.models.crop_base_model import CropBaseModel
from project.custom_layers.sampling import Sampling
import tensorflow as tf


class Crop3Test(CropBaseModel):

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
        return super().mix(source1_gt, source2_gt, mix_params)

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

    def filter(self, filter_1 ,mixed_input, alpha, bias, slope,delta):

        self._debug_image("Mezcla", mixed_input)
        self._debug_image("Estimacion de la otra fuente", filter_1)
        x_mix_filter_1 =  2 * mixed_input - filter_1
        self._debug_image("2 * mezcla - otra fuente", x_mix_filter_1)

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
    
        x_mix_filter_1 = 2 * mixed_input * mask_source1
        self._debug_image("Imagen enmascarada antes de corregir", x_mix_filter_1)
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

        x_mix_filter_1 = x_mix_filter_1 -(x_mix_filter_1-cvae_output)*delta
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
        j = self.iteration
        self._debug_source = 1
        reconstructed_source1, mask_source1, predictions_1 = (
            self.filter(self.source2_estimation,self.mixed_input, alpha_2, bias, slope,delta)
        )

        self.source1_estimation = reconstructed_source1
        self.mask1 = mask_source1
        self.predictions1 = predictions_1
        self.model_params["alpha_2"] = alpha_2 * beta

        x__x = (self.source1_estimation + self.source2_estimation) / 2
        x__x_e = x__x - self.mixed_input
        
        self.source1_estimation = self.source1_estimation - (x__x_e * gamma)
        self._debug_image("Correccion gamma", self.source1_estimation)

        self.source1_estimation = tf.clip_by_value(self.source1_estimation, 0, 1)
        self._debug_image("Estimacion final", self.source1_estimation)
        self._debug_source = 2
        reconstructed_source2, mask_source2, predictions_2 = (
            self.filter(reconstructed_source1 ,self.mixed_input, alpha_1, bias, slope,delta)
        )

        self.source2_estimation = reconstructed_source2
        self.mask2 = mask_source2
        self.predictions2 = predictions_2
        self.model_params["alpha_1"] = alpha_1 * beta



        x__x = (self.source1_estimation + self.source2_estimation) / 2
        x__x_e = x__x - self.mixed_input

        self.source2_estimation = self.source2_estimation - (x__x_e * gamma)
        self._debug_image("Correccion gamma", self.source2_estimation)


        self.source2_estimation = tf.clip_by_value(self.source2_estimation, 0, 1)
        self._debug_image("Estimacion final", self.source2_estimation)
