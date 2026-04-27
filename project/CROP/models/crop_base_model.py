import numpy as np
from project.graphics.graphics import Graphics
from project.metrics.metrics import Metrics
from abc import abstractmethod, ABC
from project.loader import Loader

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
            self.cvae = Loader.cvae(model_name=cvae_name)

        # Caso 3: parámetros clásicos
        elif lat is not None and inter is not None and dataset is not None:
            self.cvae = Loader.cvae(
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
            gen1=self.source1_estimation, gen2=self.source2_estimation
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
    show_all_curves=False,
    show_acc_curve=False,
    show_ssim_curve=False,
    show_psnr_curve=False,
    curves_save_path=None,
):

        # -------------------- params --------------------
        self.model_params = {**self.default_params.copy(), **(params or {})}

        # -------------------- mix --------------------
        self.mix(source1_gt, source2_gt, self.model_params)

        # -------------------- flags --------------------
        curves_requested = (
            show_all_curves or show_acc_curve or show_ssim_curve or show_psnr_curve
        )

        # -------------------- iteraciones --------------------
        if curves_requested:
            curves = self._collect_iteration_curves(
                source1_gt=source1_gt,
                source2_gt=source2_gt,
                source1_cond=source1_labels,
                source2_cond=source2_labels,
                iterations=iterations,
                track_acc=show_acc_curve or show_all_curves,
                track_ssim=show_ssim_curve or show_all_curves,
                track_psnr=show_psnr_curve or show_all_curves,
                track_all=show_all_curves,
            )
        else:
            for _ in range(iterations):
                self.decode()
            curves = None

        # -------------------- predicciones finales --------------------
        if self.predictions1 is None:
            self.predictions1 = self.predictor.predict(
                self.source1_estimation, verbose=False
            )

        if self.predictions2 is None:
            self.predictions2 = self.predictor.predict(
                self.source2_estimation, verbose=False
            )

        # -------------------- métricas finales --------------------
        metrics = self._compute_all_metrics(
            source1_gt, source2_gt, source1_labels, source2_labels
        )

        # -------------------- resultado --------------------
        result = {
            "predictions_1": self.predictions1,
            "predictions_2": self.predictions2,
            "model_params": self.model_params,
            **{
                k: metrics[k]
                for k in (
                    "recon_bpsnr",
                    "ssim",
                    "mask_bpsnr",
                    "acc_at_least_one",
                    "acc_both",
                )
            },
        }

        # 🔥 agregar curvas al resultado
        if curves is not None:
            result.update({
                "acc_both_plot": curves.get("acc_both", []),
                "ssim_plot": curves.get("ssim", []),
                "psnr_plot": curves.get("recon_bpsnr", []),
            })

        # -------------------- imagen --------------------
        if show_image:
            self.graphicator.complete_plot(
                self.mixed_input,
                source1_gt,
                source2_gt,
                source1_labels,
                source2_labels,
                self.source1_estimation,
                self.source2_estimation,
                self.mask1,
                self.mask2,
                self.predictions1,
                self.predictions2,
                metrics["best_prediction_source1"],
                model_params=self.model_params,
                metrics=metrics,
                save_path=save_path,
                class_labels=labels,
            )

        # -------------------- curvas --------------------
        if curves_requested:
            plot_data = {}

            if show_acc_curve or show_all_curves:
                plot_data["accuracy"] = {
                    "acc_both": curves["acc_both"],
                }

            if show_ssim_curve or show_all_curves:
                plot_data["ssim"] = {
                    "ssim": curves["ssim"],
                }

            if show_psnr_curve or show_all_curves:
                plot_data["psnr"] = {
                    "recon_psnr": curves["recon_bpsnr"],
                }

            self.graphicator.curves_plot(
                plot_data,
                model_params=self.model_params,
                title=f"{self.name}",
                save_path=curves_save_path,
            )

        # -------------------- print --------------------
        if show_metrics:
            print("\n======= PARÁMETROS DEL MODELO =====================")
            self._print_named_table(self.model_params)

            print("\n============= MÉTRICAS ===========================")
            self._print_named_table(metrics)
            print("==================================================\n")

        # -------------------- reset --------------------
        self.model_params = self.default_params.copy()

        return result

    def _params_text(self, model_params):
        parts = []
        for k, v in model_params.items():
            if isinstance(v, (int, float, np.floating)):
                parts.append(f"{k}={float(v):.3f}")
            else:
                parts.append(f"{k}={v}")
        return " | ".join(parts)

    def _collect_iteration_curves(
        self,
        source1_gt,
        source2_gt,
        source1_cond,
        source2_cond,
        iterations,
        track_acc=False,
        track_ssim=False,
        track_psnr=False,
        track_all=False,
    ):
        curves = {
            "acc_at_least_one": [],
            "acc_both": [],
            "ssim": [],
            "recon_bpsnr": [],
            "mask_bpsnr": [],
        }

        for _ in range(iterations):
            self.decode()

            if track_acc:
                self.predictions1 = self.predictor.predict(self.source1_estimation, verbose=False)
                self.predictions2 = self.predictor.predict(self.source2_estimation, verbose=False)

                acc_at_least_one, acc_both = self.metrics_cal.accuracys(
                    gt1=source1_cond,
                    gt2=source2_cond,
                    p1=self.predictions1,
                    p2=self.predictions2,
                )
                curves["acc_both"].append(acc_both)

            if track_ssim or track_all:
                ssim_mean, _ = self.metrics_cal.batched_ssim(
                    gt1=source1_gt,
                    gt2=source2_gt,
                    gen1=self.source1_estimation,
                    gen2=self.source2_estimation,
                )
                curves["ssim"].append(ssim_mean)

            if track_psnr or track_all:
                recon_psnr_mean, _ = self.metrics_cal.batched_psnr(
                    gt1=source1_gt,
                    gt2=source2_gt,
                    gen1=self.source1_estimation,
                    gen2=self.source2_estimation,
                )
                curves["recon_bpsnr"].append(recon_psnr_mean)

                if track_all:
                    mask_psnr_mean, _ = self.metrics_cal.batched_psnr(
                        gt1=source1_gt,
                        gt2=source2_gt,
                        gen1=self.mask1,
                        gen2=self.mask2,
                    )
                    curves["mask_bpsnr"].append(mask_psnr_mean)

        return curves

    def metric_curve(
        self,
        source1_gt,
        source2_gt,
        source1_cond,
        source2_cond,
        iterations=3,
        params=None,
        name=None,
        show_acc_curve=False,
        show_ssim_curve=False,
        show_psnr_curve=False,
        show_all_curves=False,
        save_path=None,
    ):
        self.model_params = {**self.default_params.copy(), **(params or {})}
        self.mix(source1_gt, source2_gt, self.model_params)

        curves = self._collect_iteration_curves(
            source1_gt=source1_gt,
            source2_gt=source2_gt,
            source1_cond=source1_cond,
            source2_cond=source2_cond,
            iterations=iterations,
            track_acc=show_acc_curve or show_all_curves,
            track_ssim=show_ssim_curve or show_all_curves,
            track_psnr=show_psnr_curve or show_all_curves,
            track_all=show_all_curves,
        )

        plot_data = {}
        if show_acc_curve or show_all_curves:
            plot_data["accuracy"] = {
                "acc_at_least_one": curves["acc_at_least_one"],
                "acc_both": curves["acc_both"],
            }
        if show_ssim_curve or show_all_curves:
            plot_data["ssim"] = {
                "ssim": curves["ssim"],
            }
        if show_psnr_curve or show_all_curves:
            plot_data["psnr"] = {
                "recon_bpsnr": curves["recon_bpsnr"],
            }
            if show_all_curves and len(curves["mask_bpsnr"]) > 0:
                plot_data["psnr"]["mask_bpsnr"] = curves["mask_bpsnr"]

        if plot_data:
            self.graphicator.curves_plot(
                plot_data,
                model_params=self.model_params,
                title=name or f"Curvas - {self.name}",
                save_path=save_path,
            )

        self.model_params = self.default_params.copy()
        return curves
    
    
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

