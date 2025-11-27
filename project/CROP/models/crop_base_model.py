from tkinter import N
import numpy as np
from project.CROP.utitls.graphics import Graphics
from project.CROP.utitls.metrics import Metrics

import project.inference.outcomes as out
import project.inference.metrics as met

from abc import abstractmethod, ABC

"""
x_mix_orig → mixed_input
(the initial mixture of both sources)

x → source1_gt
(ground truth image of source 1)

x_1 → source2_gt
(ground truth image of source 2)

y → source1_cond
(conditioning vector/label for source 1)

y_1 → source2_cond
(conditioning vector/label for source 2)

x_mix_filtrado_1 → reconstructed_source1
(filtered estimate of source 1)

x_mix_filtrado_2 → reconstructed_source2
(filtered estimate of source 2)

x_decoded_1 → mask_source1
(decoder mask/activation applied to mixture for source 1)

x_decoded_2 → mask_source2
(decoder mask/activation applied to mixture for source 2)

x__x → init_placeholder
(zeros tensor used for initialization)

x_best_predicted_1 → best_prediction_source1
(final refined reconstruction of source 1 after evaluation)
"""


class CropBaseModel(ABC):
    """
    Esto es una clase abstracta. Para crear un modelo nuevo hay que crear una clase nueva que hereder de esta
    y definir las funciones filter y decode.
    
    """
    def __init__(self, cvae, predictor, model_params=None, **kwargs):
        #auto enconder
        self.cvae = cvae
        #predictor
        self.predictor = predictor
        #parametos por defecto
        default_params = {
            "alpha_1": -2,
            "alpha_2": -22,
            "bias": 0.22,
            "slope": 22,
            "gamma": 0.33,
            "alpha_mix": 0.5,
            "beta": 1,
        }

        self.model_params = {**default_params, **(model_params or {})}
        #modulos de calculos/graficos
        self.graphicator = Graphics
        self.metrics_cal = Metrics
        #datos de la úlitma pasada
        self.mix_input=None
        self.mask1=None
        self.mask2=None
        self.source1_estimation=None
        self.source2_estimation=None
        self.predictions1=None
        self.predictions2=None
        ##metricas
        self.metrics={}
        
        

    @abstractmethod
    def filter(self,mixed_input, filter_1):
        
        """
        Input:mixed_input, filter_1, params

        filter_1: Cambiar nombre. estimación de alguna fuente 
        mixed_input: Input original

        Return: x_mix_filter_1, mask_source1, condition_encoder

        x_mix_filter_1: Nuevo estimación de la fuente que no se usa de input. Tiene que tener las mismas dimensiones que filer_1
        mask_source1: Mascara de la fuente nueva. Tiene que tener las mismas dimensiones que filer_1
        condition_encoder: Predicción de la clase de la fuente que se estimó. Las dimensiones [n_imagenes,n_clases]
        """

        pass

    @abstractmethod
    def decode(self):
        """
        Input:
        mixed_input: Mezcla original.
        reconstructed_source1: Estimación de la fuente 1. 
        reconstructed_source2: Estimación de la fuente 2.
        params: Parámetros opcionales para la decodificación. 

        Return:

        mask_source1: Actualización Máscara 1. Dim: Igual que las mezclas
        mask_source2: Actualización Máscara 2. Dim: Igual que las mezclas
        reconstructed_source1: Actualización de la reconstrucción de la fuente 1 Dim: Igual que las mezclas
        reconstructed_source2: Actualización de la reconstrucción de la fuente 2 Dim: Igual que las mezclas
        predictions_1: Predicción de la clase 1. [n_imagenes, n_clases]
        predictions_2: Predicción de la clase 2. [n_imagenes, n_clases]
        

        """
        pass

    def mix(self,source1_gt,source2_gt,mix_params):
        
        average_image = mix_params["alpha_mix"] * source1_gt.astype(np.float32) + (
            1 - mix_params["alpha_mix"]
        ) * source2_gt.astype(np.float32)
        
        self.mixed_input = average_image
        self.mask1 = average_image
        self.mask2 = average_image
        self.source1_estimation = average_image
        self.source2_estimation = average_image

        return average_image


    def unmix(
        self,
        source1_gt,
        source2_gt,
        source1_labels,
        source2_labels,
        iterations=3,
        show_image=False,
        save_path=None,
        params=None,
    ):
                
        # combinar defaults con parámetros recibidos
        self.model_params = {**self.model_params, **(params or {})}

        self.mix(source1_gt,source2_gt,self.model_params)

        for _ in range(iterations):
            
             self.decode()

        bpsnr_mean_estimation, bpsnr_std_estimation = self.metrics_cal.batched_psnr(gt1=source1_gt,
                                                              gt2=source2_gt,
                                                              gen1=self.source1_estimation,
                                                              gen2=self.source2_estimation)
        
        bpsnr_mean_mask, bpsnr_std_mask = self.metrics_cal.batched_psnr(gt1=source1_gt,
                                                              gt2=source2_gt,
                                                              gen1=self.mask1,
                                                              gen2=self.mask2)

        acc_at_least_one, acc_both = self.metrics_cal.accuracys(gt1=source1_labels, 
                                                                gt2=source2_labels,
                                                                p1=self.predictions1,
                                                                p2=self.predictions2)
        print("accs:",  acc_at_least_one, acc_both )

        best_prediction_source1 = self.metrics_cal.best_predicctions(source1_gt,source2_gt,source1_labels,
    source2_labels)

        if show_image:
            self.graphicator.complete_plot(
                self.mixed_input,
                source1_gt,
                source2_gt,
                self.source1_estimation,
                self.source2_estimation,
                self.mask1,
                self.mask2,
                best_prediction_source1,
                bias=self.model_params["bias"],
                slope=self.model_params["slope"],
                title="",
                bpsnr=bpsnr_mean_estimation,  # mean value
                acc_at_least_one=acc_at_least_one,
                acc_both=acc_both,
                save_path=save_path,
            )

        return { # devolver 2 dicionarios, uno de metricas y otro de "imagenes"
            "bpsnr": bpsnr_mean_estimation,
            "bpsnr_d": bpsnr_mean_mask,
            "predictions_1": self.predictions1,
            "predictions_2": self.predictions2,
            "acc_at_least_one": acc_at_least_one,
            "acc_both": acc_both,
        }

    def acc_curve(
        self,
        source1_gt,
        source2_gt,
        source1_cond,
        source2_cond,
        iterations=3,
        params=None,
        name=None
    ):

       # combinar defaults con parámetros recibidos
        params = {**self.model_params, **(params or {})}

        mixed_input, mask_source1, mask_source2, reconstructed_source1, reconstructed_source2 = self.mix(source1_gt,source2_gt,params)

        acc_at_least_one_plot = []
        acc_both_plot = []

        for _ in range(iterations):
            (
                mask_source1,
                mask_source2,
                reconstructed_source1,
                reconstructed_source2,
                predictions_1,
                predictions_2,
            ) = self.decode(
                mixed_input,
                reconstructed_source1,
                reconstructed_source2,
                params
            )

            y_predicted_s1_recon = self.predictor(
                reconstructed_source1, training=False, verbose=0
            )
            y_predicted_s2_recon = self.predictor(
                reconstructed_source2, training=False, verbose=0
            )

            acc_at_least_one, acc_both = met.accuracys(
                gt1=source1_cond, 
                gt2=source2_cond,
                p1=y_predicted_s1_recon,
                p2=y_predicted_s2_recon
            )

            acc_at_least_one_plot.append(acc_at_least_one)
            acc_both_plot.append(acc_both)

        self.graphicator.acc_plot(acc_at_least_one_plot,acc_both_plot,name)

        return {
            "acc_at_least_one_plot": acc_at_least_one_plot,
            "acc_both_plot": acc_both_plot,
        }
