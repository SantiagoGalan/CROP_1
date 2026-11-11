"""Estructura experimental: Crop3Test seguido de un procesamiento por definir."""

from project.CROP.models.crop3_test import Crop3Test


class Crop4Test(Crop3Test):
    """Separa durante 20 iteraciones y luego entrega las dos estimaciones.

    El segundo procesamiento todavia no esta implementado: devuelve ambas
    imagenes sin cambios. Se ejecuta una sola vez al terminar la iteracion
    separation_iterations, despues de todo el decode de Crop3Test.
    Las llamadas posteriores conservan las estimaciones finales. Si unmix
    recibe menos iteraciones que el umbral, no llega a la segunda etapa.
    Reutiliza la base, filter, unmix y funciones de debug existentes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_params = {
            **self.default_params,
            "delta": 0.0,
            "separation_iterations": 20,
        }
        self.model_params = self.default_params.copy()
        self.processing_complete = False

    def mix(self, source1_gt, source2_gt, mix_params):
        steps = mix_params.get("separation_iterations", 20)
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("separation_iterations debe ser un entero positivo")
        self.processing_complete = False
        return super().mix(source1_gt, source2_gt, mix_params)

    def process_estimations(self, source1_estimation, source2_estimation):
        """Punto de extension para el segundo procesamiento, pendiente.

        Recibe y devuelve ambas estimaciones con su forma de batch original.
        La mezcla y las mascaras estan disponibles en self.mixed_input,
        self.mask1 y self.mask2. Por ahora no modifica ningun pixel.
        """
        return source1_estimation, source2_estimation

    def decode(self):
        if self.processing_complete:
            return
        super().decode()
        if self.iteration + 1 >= self.model_params["separation_iterations"]:
            self.source1_estimation, self.source2_estimation = self.process_estimations(
                self.source1_estimation, self.source2_estimation
            )
            # unmix debe evaluar las imagenes devueltas por la segunda etapa.
            self.predictions1 = None
            self.predictions2 = None
            self.processing_complete = True
