"""Crop3Test seguido de un ajuste conjunto sin utilizar fuentes originales."""

import math

import tensorflow as tf

from project.CROP.models.crop3_test import Crop3Test


class Crop4Test(Crop3Test):
    """Separa durante 20 iteraciones y refina una vez las estimaciones.

    Segunda etapa: proyeccion a la mezcla, anclaje segun confianza,
    suavidad de las correcciones y continuidad sensible a los bordes.
    El mapa de sospecha es una heuristica, no una medida del error real.
    Si unmix pide menos iteraciones que el umbral, no ejecuta esta etapa.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_params = {
            **self.default_params,
            "delta": 0.0,
            "separation_iterations": 20,
            "refinement_iterations": 80,
            "refinement_step": 0.05,
            "refinement_tolerance": 1e-5,
            "refinement_correction_weight": 1.0,
            "refinement_continuity_weight": 0.2,
            "refinement_signal_scale": 0.05,
            "refinement_edge_scale": 0.1,
        }
        self.model_params = self.default_params.copy()
        self._reset_processing()

    def _reset_processing(self):
        self.processing_complete = False
        self.raw_masked_sources = {}
        self.refinement_maps = {}
        self.refinement_report = {}
        self.refinement_loss_history = []
        self.pre_refinement_estimations = None

    def mix(self, source1_gt, source2_gt, mix_params):
        for name in ("separation_iterations", "refinement_iterations"):
            value = mix_params.get(name, self.default_params[name])
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} debe ser un entero positivo")
        alpha = mix_params.get("alpha_mix", 0.5)
        if not 0 < alpha < 1:
            raise ValueError("alpha_mix debe estar estrictamente entre 0 y 1")
        for name in ("refinement_step", "refinement_signal_scale", "refinement_edge_scale",
                     "refinement_tolerance", "refinement_correction_weight",
                     "refinement_continuity_weight"):
            value = mix_params.get(name, self.default_params[name])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} debe ser finito y no negativo")
            if name in ("refinement_step", "refinement_signal_scale", "refinement_edge_scale") and value == 0:
                raise ValueError(f"{name} debe ser positivo")
        self._reset_processing()
        return super().mix(source1_gt, source2_gt, mix_params)

    def subtract_saturation_border_mean(self, masked_input, object_threshold=0.05):
        # Conservar la evidencia perdida por el clip sin alterar Crop3Test.
        self.raw_masked_sources[self._debug_source] = tf.identity(masked_input)
        return super().subtract_saturation_border_mean(masked_input, object_threshold)

    @staticmethod
    def _image4d(value):
        value = tf.convert_to_tensor(value)
        shape = tf.shape(value)
        if value.shape.rank == 2:
            side = tf.cast(tf.sqrt(tf.cast(shape[1], tf.float64)), tf.int32)
            tf.debugging.assert_equal(side * side, shape[1])
            return tf.reshape(value, [shape[0], side, side, 1])
        if value.shape.rank == 3:
            return value[..., None]
        if value.shape.rank != 4:
            raise ValueError("Se esperaba un batch de imagenes de rango 2, 3 o 4")
        return value

    @staticmethod
    def _project(x, y, mixed, alpha):
        """Par mas cercano en [0,1] que cumple a*x+(1-a)*y=mezcla."""
        a = tf.cast(alpha, x.dtype)
        b = 1 - a
        lower = tf.maximum(0.0, (mixed - b) / a)
        upper = tf.minimum(1.0, mixed / a)
        x = tf.clip_by_value(x + a * (mixed - a*x - b*y) / (a*a + b*b), lower, upper)
        y = tf.clip_by_value((mixed - a*x) / b, 0, 1)
        return x, y

    @staticmethod
    def _differences(image):
        return image[:, 1:, :, :] - image[:, :-1, :, :], image[:, :, 1:, :] - image[:, :, :-1, :]

    def _suspicion(self, image, residual, raw, overlap):
        scale = self.model_params["refinement_signal_scale"]
        local_mean = tf.nn.avg_pool2d(image, 3, 1, "SAME")
        # Variacion local alta no prueba un error: se pondera junto a otras senales.
        anomaly = tf.abs(image - local_mean)
        saturation = tf.cast(raw > 1, image.dtype)
        score = (0.5 * residual / (residual + scale)
                 + 0.3 * saturation + 0.2 * anomaly / (anomaly + scale))
        score *= 0.5 + 0.5 * overlap
        return tf.nn.avg_pool2d(score, 3, 1, "SAME")

    def process_estimations(self, source1_estimation, source2_estimation):
        """Optimiza solo con mezcla, mascaras y estimaciones de la primera etapa.

        La igualdad de mezcla es obligatoria incluso en zonas confiables; su
        conservacion y la continuidad son penalizaciones, no garantias de
        recuperar las fuentes reales ni de eliminar todos los bordes abruptos.
        """
        p = self.model_params
        shape = tf.shape(source1_estimation)
        x0 = self._image4d(source1_estimation)
        y0 = self._image4d(source2_estimation)
        mixed = self._image4d(self.mixed_input)
        tf.debugging.assert_all_finite(mixed, "La mezcla contiene valores no finitos")
        tf.debugging.assert_greater_equal(mixed, tf.cast(0, mixed.dtype))
        tf.debugging.assert_less_equal(mixed, tf.cast(1, mixed.dtype))
        a = p["alpha_mix"]
        residual = tf.abs(a*x0 + (1-a)*y0 - mixed)
        overlap = tf.minimum(self._image4d(self.mask1), self._image4d(self.mask2))
        raw1 = self._image4d(self.raw_masked_sources.get(1, source1_estimation))
        raw2 = self._image4d(self.raw_masked_sources.get(2, source2_estimation))
        suspicion1 = self._suspicion(x0, residual, raw1, overlap)
        suspicion2 = self._suspicion(y0, residual, raw2, overlap)
        self.pre_refinement_estimations = (tf.identity(source1_estimation), tf.identity(source2_estimation))
        self.refinement_maps = {
            "mixture_error": tf.reshape(residual, shape),
            "suspicion1": tf.reshape(suspicion1, shape),
            "suspicion2": tf.reshape(suspicion2, shape),
            "confidence1": tf.reshape(1-suspicion1, shape),
            "confidence2": tf.reshape(1-suspicion2, shape),
        }
        anchors = (1 - 0.9*suspicion1, 1 - 0.9*suspicion2)
        edge_weights = []
        for original, suspicion in ((x0, suspicion1), (y0, suspicion2)):
            dy, dx = self._differences(original)
            sy = (suspicion[:, 1:] + suspicion[:, :-1]) / 2
            sx = (suspicion[:, :, 1:] + suspicion[:, :, :-1]) / 2
            edge_weights.append((
                sy * tf.exp(-tf.square(dy / p["refinement_edge_scale"])),
                sx * tf.exp(-tf.square(dx / p["refinement_edge_scale"])),
            ))

        def objective(x, y):
            loss = tf.reduce_sum(anchors[0]*tf.square(x-x0) + anchors[1]*tf.square(y-y0)) / 2
            for current, original, edges in ((x, x0, edge_weights[0]), (y, y0, edge_weights[1])):
                cy, cx = self._differences(current-original)
                iy, ix = self._differences(current)
                loss += p["refinement_correction_weight"] * (tf.reduce_sum(cy*cy) + tf.reduce_sum(cx*cx)) / 2
                loss += p["refinement_continuity_weight"] * (
                    tf.reduce_sum(edges[0]*iy*iy) + tf.reduce_sum(edges[1]*ix*ix)
                ) / 2
            return loss

        x, y = self._project(x0, y0, mixed, a)
        # Cota del Hessiano de diferencias en cuatro vecinos: paso estable.
        step = min(p["refinement_step"], 0.9 / (
            1 + 8*p["refinement_correction_weight"] + 8*p["refinement_continuity_weight"]
        ))
        self.refinement_loss_history = [float(objective(x, y))]
        for iteration in range(p["refinement_iterations"]):
            with tf.GradientTape() as tape:
                tape.watch([x, y])
                loss = objective(x, y)
            gx, gy = tape.gradient(loss, [x, y])
            new_x, new_y = self._project(x-step*gx, y-step*gy, mixed, a)
            change = tf.maximum(tf.reduce_max(tf.abs(new_x-x)), tf.reduce_max(tf.abs(new_y-y)))
            x, y = new_x, new_y
            self.refinement_loss_history.append(float(objective(x, y)))
            for source, image in ((1, x), (2, y)):
                self._debug_source = source
                self._debug_image(f"Ajuste conjunto {iteration + 1}", tf.reshape(image, shape))
            if float(change) <= p["refinement_tolerance"]:
                break
        self.refinement_report = {
            "iterations": len(self.refinement_loss_history)-1,
            "initial_objective": self.refinement_loss_history[0],
            "final_objective": self.refinement_loss_history[-1],
            "mixture_max_error_before": float(tf.reduce_max(residual)),
            "mixture_max_error_after": float(tf.reduce_max(tf.abs(a*x + (1-a)*y - mixed))),
            "step": step,
        }
        return tf.reshape(x, shape), tf.reshape(y, shape)

    def show_refinement_maps(self):
        """Visualiza las senales de confianza de la primera imagen del batch."""
        import matplotlib.pyplot as plt

        if not self.refinement_maps:
            print("Ejecutar la segunda etapa antes de mostrar sus mapas.")
            return
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, name, title in zip(axes, ("mixture_error", "suspicion1", "suspicion2"),
                                   ("Error de mezcla previo", "Sospecha fuente 1", "Sospecha fuente 2")):
            image = self._image4d(self.refinement_maps[name])[0, ..., 0].numpy()
            plot = ax.imshow(image, cmap="magma", vmin=0, vmax=1 if name != "mixture_error" else None)
            ax.set_title(title)
            ax.axis("off")
            fig.colorbar(plot, ax=ax, fraction=0.046)
        fig.tight_layout()
        plt.show()
        plt.close(fig)

    def decode(self):
        if self.processing_complete:
            return
        super().decode()
        if self.iteration + 1 >= self.model_params["separation_iterations"]:
            self.source1_estimation, self.source2_estimation = self.process_estimations(
                self.source1_estimation, self.source2_estimation
            )
            self.predictions1 = None
            self.predictions2 = None
            self.processing_complete = True
