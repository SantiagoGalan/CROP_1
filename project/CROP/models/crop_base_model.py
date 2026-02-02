import numpy as np
from project.graphics.graphics import Graphics
from project.metrics.metrics import Metrics
from abc import abstractmethod, ABC
from project.loader import load

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
    def __init__(
        self,
        predictor,
        cvae=None,
        *,
        lat=None,
        inter=None,
        dataset=None,
        cvae_name=None,
    ):

        self.predictor = predictor

        # Caso 1: CVAE ya cargado
        if cvae is not None:
            self.cvae = cvae

        # Caso 2: nombre base del CVAE
        elif cvae_name is not None:
            self.cvae = load.cvae(model_name=cvae_name)

        # Caso 3: parámetros clásicos
        elif lat is not None and inter is not None and dataset is not None:
            self.cvae = load.cvae(
                lat=lat,
                inter=inter,
                dataset=dataset,
            )

        else:
            raise ValueError(
                "Debes pasar un cvae, cvae_name o (lat, inter, dataset)"
            )

        #parametos por defecto
        self.default_params = {
            "alpha_1": -2,
            "alpha_2": -22,
            "bias": 0.22,
            "slope": 22,
            "gamma": 0.33,
            "alpha_mix": 0.5,
            "beta": 1,
        }
        self.name = self.cvae.name
        self.model_params = self.default_params
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
        
        self.mixed_input = (average_image)
        self.mask1 = (average_image)
        self.mask2 = (average_image)
        self.source1_estimation = (average_image)
        self.source2_estimation = (average_image)

        return average_image

    def _compute_all_metrics(self, gt1, gt2, lbl1, lbl2):

        bpsnr_mean_est, bpsnr_std_est = self.metrics_cal.batched_psnr(
            gt1=gt1, gt2=gt2,
            gen1=self.source1_estimation, gen2=self.source2_estimation
        )

        bpsnr_mean_mask, bpsnr_std_mask = self.metrics_cal.batched_psnr(
            gt1=gt1, gt2=gt2,
            gen1=self.mask1, gen2=self.mask2
        )

        ssim_mean, ssim_std = self.metrics_cal.batched_ssim(
            gt1=gt1, gt2=gt2,
            gen1=self.mask1, gen2=self.mask2
        )

        acc_at_least_one, acc_both = self.metrics_cal.accuracys(
            gt1=lbl1, gt2=lbl2,
            p1=self.predictions1, p2=self.predictions2
        )

        best_prediction_source1 = self.metrics_cal.best_predicctions(
            gt1, gt2, lbl1, lbl2
        )

        return {
            "recon_bpsnr": (bpsnr_mean_est, bpsnr_std_est),
            "mask_bpsnr": (bpsnr_mean_mask, bpsnr_std_mask),
            "ssim": (ssim_mean, ssim_std),
            "acc_at_least_one": acc_at_least_one,
            "acc_both": acc_both,
            "best_prediction_source1": best_prediction_source1,
        }

    def _format_dict_for_printing(self, data):

        formatted = {}
        for k, v in data.items():

            if k in ("predictions_1", "predictions_2", "best_prediction_source1"):
                continue

            if isinstance(v, tuple) and len(v) == 2: #si hay 2 metricas con media y std
                mean, std = v
                formatted[k] = f"mean: {mean:.3f}  std:{std:.3f}"

            elif isinstance(v, (int, float)):
                formatted[k] = f"{v:.3f}"

            else:
                formatted[k] = str(v)

        return formatted


    def _print_named_table(self, data):
        data = self._format_dict_for_printing(data)
        max_key_len = max(len(k) for k in data.keys())

        for key, value in data.items():
            print(f"{key.ljust(max_key_len)} : {value}")

    def unmix(
        self,
        source1_gt,
        source2_gt,
        source1_labels,
        source2_labels,
        iterations=10,
        show_image=False,
        show_metrics=True,
        save_path=None,
        params=None,
        labels=None,
    ):

        # Mezclar parámetros por defecto + overrides
        self.model_params = {**self.model_params, **(params or {})}

        # Mezcla 
        self.mix(source1_gt, source2_gt, self.model_params)

        # Decodificación 
        for _ in range(iterations):
            self.decode()

        # Cálculo de métricas
        metrics = self._compute_all_metrics(
            source1_gt, source2_gt, source1_labels, source2_labels
        )

        # Armar diccionario final del resultado
        result = {
            "predictions_1": self.predictions1,
            "predictions_2": self.predictions2,
            "model_params": self.model_params,
            **{k: metrics[k] for k in ("recon_bpsnr", "ssim", "mask_bpsnr", "acc_at_least_one", "acc_both")}
        }

        # Mostrar imagen con métricas
        if show_image:
            self.graphicator.complete_plot(
                self.mixed_input,
                source1_gt, source2_gt,
                source1_labels,source2_labels,
                self.source1_estimation, self.source2_estimation,
                self.mask1, self.mask2,
                self.predictions1,self.predictions2,
                metrics["best_prediction_source1"],
                model_params=self.model_params,
                metrics=metrics,
                #title=f"Modelo: {self.name}",
                save_path=save_path,
                class_labels=labels,
            )

        # Mostrar tabla
        if show_metrics:
            print("\n======= PARÁMETROS DEL MODELO =====================")
            self._print_named_table(self.model_params)

            print("\n============= MÉTRICAS ===========================")
            self._print_named_table(metrics)
            print("==================================================\n")

        # reset a defaults
        self.model_params = self.default_params

        return result

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

        # seteo de parametros
        self.model_params = {**self.model_params, **(params or {})}

        # Mezcla 
        self.mix(source1_gt, source2_gt, self.model_params)

        acc_at_least_one_plot = []
        acc_both_plot = []

             # Decodificación 
        for _ in range(iterations):
            self.decode()
            prediction1 = self.predictor.predict(self.source1_estimation,verbose=False)
            prediction2 = self.predictor.predict(self.source2_estimation,verbose=False)
            
            #calculo acc es cada iteración para el grafico
            acc_at_least_one, acc_both = self.metrics_cal.accuracys(
                    gt1=source1_cond, gt2=source2_cond,
                    p1=prediction1, p2=prediction2
                )


            acc_at_least_one_plot.append(acc_at_least_one)
            acc_both_plot.append(acc_both)

        self.graphicator.acc_plot(acc_at_least_one_plot,acc_both_plot,name,params=params)
      
        return {
            "acc_at_least_one_plot": acc_at_least_one_plot,
            "acc_both_plot": acc_both_plot,
        }
    

    def psnr_curve(
            self,
            source1_gt,
            source2_gt,
            source1_cond,
            source2_cond,
            iterations=3,
            params=None,
            name=None
        ):

        # seteo de parametros
        self.model_params = {**self.model_params, **(params or {})}

        # Mezcla 
        self.mix(source1_gt, source2_gt, self.model_params)

        psnr_mean_resuts = []
        psnr_std_resuts = []

             # Decodificación 
        for _ in range(iterations):
            self.decode()
            psnr_mean, psnr_std = self.metrics_cal.batched_psnr(
                    gt1=source1_gt, gt2=source2_gt,
                    gen1=self.source1_estimation, gen2=self.source1_estimation
                )

            psnr_mean_resuts.append(psnr_mean)
            psnr_std_resuts.append(psnr_std)

        #self.graphicator.acc_plot(psnr_mean,psnr_std,name,params=params)
      
        return {
            "psnr_mean": psnr_mean_resuts,
            "psnr_std": psnr_std_resuts,
        }
    

    def ssim_curve(
            self,
            source1_gt,
            source2_gt,
            source1_cond,
            source2_cond,
            iterations=3,
            params=None,
            name=None
        ):

        # seteo de parametros
        self.model_params = {**self.model_params, **(params or {})}

        # Mezcla 
        self.mix(source1_gt, source2_gt, self.model_params)

        ssim_mean_resuts = []
        ssim_std_resuts = []

             # Decodificación 
        for _ in range(iterations):
            self.decode()
            ssim_mean, ssim_std = self.metrics_cal.batched_ssim(
                    gt1=source1_gt, gt2=source2_gt,
                    gen1=self.source1_estimation, gen2=self.source1_estimation
                )

            ssim_mean_resuts.append(ssim_mean)
            ssim_std_resuts.append(ssim_std)

        #self.graphicator.acc_plot(ssim_mean,ssim_std,name,params=params)
      
        return {
            "ssim_mean": ssim_mean_resuts,
            "ssim_std": ssim_std_resuts,
        }
    
    
    def reconstruction_by_condition(self,x_input):
        """
        Muestra cómo se reconstruye una imagen de entrada bajo las 10 condiciones posibles (0 a 9).
        """
        x_input = np.expand_dims(x_input, axis=0)

        x_repeated = np.repeat(x_input, repeats=10, axis=0)

        one_hot_conditions = np.eye(10)  # I(10, 10)

        z_mean, z_log_var, z = self.cvae.encoder.predict([x_repeated, one_hot_conditions])

        recons = self.cvae.decoder.predict([z, one_hot_conditions])
        
        Graphics.reconstruction_by_condition(recons)

