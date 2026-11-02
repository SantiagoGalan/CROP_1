"""Modelo experimental de corrección localizada y compresión local de intensidades."""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from project.CROP.models.crop_base_model import CropBaseModel
from project.custom_layers.sampling import Sampling


class crop3test(CropBaseModel):
    """Copia experimental de Crop3EstimationCorrection; no modifica el modelo original."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_params.update({
            "delta": 0.3,
            "mode": "local_compression",
            "window": 5,
            "object_threshold": 0.5,
            "local_k": 2.0,
        })
        self.debug = False
        self.debug_max_iteration = 0  # Primera iteración (el bucle empieza en cero).
        self.debug_max_images = 1
        self.image_shape = (28, 28)

    def show_intermediates(self, stages, title=""):
        """Muestra un panel por imagen, con escala fija y rango numérico real."""
        if not self.debug or self.iteration > self.debug_max_iteration:
            return
        first = np.asarray(next(iter(stages.values())))
        for index in range(min(self.debug_max_images, len(first))):
            fig, axes = plt.subplots(1, len(stages), figsize=(3 * len(stages), 3),
                                     squeeze=False)
            for ax, (name, tensor) in zip(axes[0], stages.items()):
                img = np.asarray(tensor)[index].reshape(self.image_shape)
                # La visualización recorta a [0,1]; los títulos muestran el rango real.
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
                ax.set_title(f"{name}\n[{img.min():.3f}, {img.max():.3f}]")
                ax.axis("off")
            fig.suptitle(f"Iteración {self.iteration} · {title} · imagen {index}")
            plt.tight_layout()
            plt.show()

    def local_compression(self, raw, mask, cvae_output):
        """Estadísticas de vecinos válidos y compresión suave con banda de transición.

        El rango local es una heurística: la máscara aproxima el objeto, no es
        una segmentación de referencia. Se conservan valores por debajo del
        umbral local; la banda superior también puede incluir píxeles < 1.
        """
        shape = tf.shape(raw)
        images = tf.reshape(raw, [-1, *self.image_shape, 1])
        objects = tf.reshape(mask, tf.shape(images)) > self.model_params["object_threshold"]
        valid = tf.cast(objects & (images < 1.0) & (images > 0.0), images.dtype)
        window = int(self.model_params["window"])
        if window < 3 or window % 2 == 0:
            raise ValueError("window debe ser impar y >= 3")

        def pool(x):
            return tf.nn.avg_pool2d(x, ksize=window, strides=1, padding="SAME")

        count = pool(valid)
        mean = tf.math.divide_no_nan(pool(images * valid), count)
        second = tf.math.divide_no_nan(pool(tf.square(images) * valid), count)
        std = tf.sqrt(tf.maximum(second - tf.square(mean), 0.0))
        # Sin vecinos válidos se usa la reconstrucción del CVAE como referencia.
        fallback = tf.reshape(cvae_output, tf.shape(images))
        mean = tf.where(count > 0, mean, fallback)
        std = tf.where(count > 0, std, tf.ones_like(std) * 0.05)
        lower = tf.clip_by_value(mean, 0.05, 0.90)
        upper = tf.minimum(1.0, tf.maximum(
            lower + 0.05, mean + self.model_params["local_k"] * std))
        width = upper - lower
        excess = tf.nn.relu(images - lower)
        compressed = lower + width * (-tf.math.expm1(-excess / width))
        # Incluye siempre raw > 1, aun si la máscara no identifica el objeto.
        affected = (objects | (images > 1.0)) & (images > lower)
        result = tf.where(affected, compressed, images)
        return (tf.reshape(result, shape), tf.reshape(lower, shape),
                tf.reshape(upper, shape))

    def filter(self, filter_1, mixed_input, alpha, bias, slope, delta):
        residual = 2 * mixed_input - filter_1
        encoder_input = tf.clip_by_value(residual, 0, 1)
        condition_encoder = self.predictor(encoder_input, training=False)
        encoded_imgs = self.cvae.encoder(
            [encoder_input, condition_encoder], training=False)
        zz_log_var = encoded_imgs[1] + alpha
        z = Sampling()((encoded_imgs[0], zz_log_var))
        cvae_output = self.cvae.decoder([z, condition_encoder], training=False)
        mask_source1 = tf.sigmoid((cvae_output - bias) * slope)
        raw = 2 * mixed_input * mask_source1
        saturated = raw > 1.0  # Antes del clip.
        mode = self.model_params["mode"]
        if not 0 <= delta <= 1:
            raise ValueError("delta debe estar entre 0 y 1")
        stages = {"Mezcla": mixed_input, "Residual": residual,
                  "Entrada CVAE": encoder_input, "Reconstrucción": cvae_output,
                  "Máscara": mask_source1, "Filtrado sin clip": raw,
                  "Zona saturada": saturated}
        if mode in ("global", "saturated"):
            filtered = tf.clip_by_value(raw, 0, 1)
            corrected = (1 - delta) * filtered + delta * cvae_output
            result = corrected if mode == "global" else tf.where(
                saturated, corrected, filtered)
        elif mode in ("local_compression", "local_compression_cvae"):
            result, lower, upper = self.local_compression(raw, mask_source1, cvae_output)
            stages.update({"Rango local inferior": lower, "Rango local superior": upper})
            if mode == "local_compression_cvae":
                result = tf.where(saturated,
                                  (1 - delta) * result + delta * cvae_output, result)
        else:
            raise ValueError(f"Modo desconocido: {mode}")
        stages["Resultado del filtro"] = result
        self.show_intermediates(stages, title=mode)
        return result, mask_source1, condition_encoder

    def decode(self):
        alpha_1 = self.model_params["alpha_1"]
        alpha_2 = self.model_params["alpha_2"]
        beta = self.model_params["beta"]
        bias = self.model_params["bias"]
        slope = self.model_params["slope"]
        gamma = self.model_params["gamma"]
        delta = self.model_params["delta"]
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

        self.source1_estimation = tf.clip_by_value(self.source1_estimation, 0, 1)
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


        self.source2_estimation = tf.clip_by_value(self.source2_estimation, 0, 1)

        self.show_intermediates({
            "Mezcla": self.mixed_input,
            "Fuente 1 después de gamma": self.source1_estimation,
            "Fuente 2 después de gamma": self.source2_estimation,
        }, title="Corrección de mezcla")
