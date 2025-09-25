import tensorflow as tf
import numpy as np
from custom_layers.Sampling import Sampling
import inference.outcomes as out
import matplotlib.pyplot as plt


class crop:
    def __init__(self, cvae, predictor, data, bias=None, slope=None, **kwargs):
        self.cvae = cvae
        self.predictor = predictor
        self.use_dataset = data
        self.unmix_metrics = {}
        self.reconstruction_metrics = {}
        self.bias = 0.22 if bias is None else bias
        self.slope = 22 if slope is None else slope
        self.beta = 1
        self.alpha_1 = -2
        self.alpha_2 = -22
        self.alpha_mix = 0.5
        self.name = cvae.name

    def best_filtered_var_sigmoid(self, x_mix_filter_2, mixed_input, alpha):
        # Paso 1: mezcla inicial filtrada
        x_mix_filter_1 = 2 * mixed_input - x_mix_filter_2
        x_mix_filter_1 = tf.clip_by_value(x_mix_filter_1, 0, 1)

        # Paso 2: predictor → condición
        condition_encoder = self.predictor.predict(x_mix_filter_1, verbose=0)
        condition_decoder_1 = condition_encoder

        # Paso 3: encoder
        encoded_imgs = self.cvae.encoder.predict([x_mix_filter_1, condition_encoder], verbose=0)
        zz_log_var = encoded_imgs[1] + alpha
        z = Sampling()((encoded_imgs[0], zz_log_var))

        # Paso 4: decoder → máscara
        mask_source1 = self.cvae.decoder.predict([z, condition_decoder_1], verbose=0)
        mask_source1 = (mask_source1 - self.bias) * self.slope
        mask_source1 = tf.sigmoid(mask_source1)

        # Paso 5: aplicar máscara a mezcla
        x_mix_filter_1 = 2 * mixed_input * mask_source1
        x_mix_filter_1 = tf.clip_by_value(x_mix_filter_1, 0, 1)

        return (x_mix_filter_1, mask_source1, condition_encoder)

    def _plot_debug_images(self, images_dict, title="", save_path=None, img_size=28):
        """
        Muestra o guarda un conjunto de imágenes con etiquetas.
        images_dict: {label: tensor/np.array}
        """
        n = len(images_dict)
        fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
        if n == 1:
            axes = [axes]

        for ax, (label, img) in zip(axes, images_dict.items()):
            if isinstance(img, tf.Tensor):
                img = img.numpy()

            # --- reshape a 28x28 ---
            if img.ndim == 1:  # vector plano
                img = img.reshape((img_size, img_size))
            elif img.ndim == 2 and img.shape[0] != img_size:  
                # ej: batch_size x features → tomar solo primera imagen
                img = img[0].reshape((img_size, img_size))
            elif img.ndim == 3 and img.shape[-1] == 1:  
                # (H, W, 1)
                img = img.squeeze(-1)
            elif img.ndim == 4:  
                # (batch, H, W, C) → tomar la primera
                img = img[0, :, :, 0] if img.shape[-1] == 1 else img[0]

            ax.imshow(img, cmap="gray")
            ax.set_title(label)
            ax.axis("off")

        plt.suptitle(title, color="darkred")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.show()


    def unmix(
        self,
        source1_gt,
        source2_gt,
        source1_cond,
        source2_cond,
        Iterations=3,
        show_image=False,
        debug_images=False,   
        save_path=None,
    ):

        average_image = self.alpha_mix * source1_gt.astype(np.float32) + (
            1 - self.alpha_mix
        ) * source2_gt.astype(np.float32)
        x_mix = average_image

        # Initialization
        mixed_input = x_mix
        mask_source1 = x_mix
        mask_source2 = x_mix
        reconstructed_source1 = x_mix
        reconstructed_source2 = x_mix
        init_placeholder = tf.zeros_like(x_mix)

        if debug_images:
            self._plot_debug_images(
                {
                    "mixed_input": mixed_input,
                    "source1_gt": source1_gt,
                    "source2_gt": source2_gt,
                },
                title="Inicialización",
            )

        for j in range(Iterations):
            reconstructed_source1, mask_source1, predictions_1 = self.best_filtered_var_sigmoid(
                reconstructed_source2, mixed_input, self.alpha_2
            )
            self.alpha_2 *= self.beta

            reconstructed_source2, mask_source2, predictions_2 = self.best_filtered_var_sigmoid(
                reconstructed_source1, mixed_input, self.alpha_1
            )
            self.alpha_1 *= self.beta

            if debug_images:
                self._plot_debug_images(
                    {
                        "mask_s1": mask_source1,
                        "mask_s2": mask_source2,
                        "recon_s1": reconstructed_source1,
                        "recon_s2": reconstructed_source2,
                    },
                    title=f"Iteración {j+1}",
                )

        # Normalización de máscaras
        eps = 1e-6
        mask_sum = tf.maximum(mask_source1 + mask_source2, eps)
        m1 = mask_source1 / mask_sum
        m2 = mask_source2 / mask_sum

        reconstructed_source1 = tf.clip_by_value(2.0 * mixed_input * m1, 0.0, 1.0)
        reconstructed_source2 = tf.clip_by_value(2.0 * mixed_input * m2, 0.0, 1.0)

        if debug_images:
            self._plot_debug_images(
                {
                    "final_mask1": m1,
                    "final_mask2": m2,
                    "final_recon_s1": reconstructed_source1,
                    "final_recon_s2": reconstructed_source2,
                },
                title="Normalización final",
            )

        # Métricas
        (
            best_prediction_source1,
            y_predicted_s1_recon,
            y_predicted_s2_recon,
            bpsnr,
            bpsnr_d,
            acc_at_least_one,
            acc_both,
        ) = out.outcomes(
            mask_source1,
            mask_source2,
            reconstructed_source1,
            reconstructed_source2,
            mixed_input,
            source1_gt,
            source2_gt,
            source1_cond,
            source2_cond,
            self.predictor,
        )

        if debug_images:
            self._plot_debug_images(
                {
                    "best_pred_s1": best_prediction_source1,
                },
                title="Predicción final",
            )

        return {
            "bpsnr": bpsnr,
            "bpsnr_d": bpsnr_d,
            "predictions_1": predictions_1,
            "predictions_2": predictions_2,
            "acc_at_least_one": acc_at_least_one,
            "acc_both": acc_both,
        }
