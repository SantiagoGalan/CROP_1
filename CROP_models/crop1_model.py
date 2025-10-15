from CROP_models.crop_base_model import CropBaseModel
#from  crop_base_model import CropBaseModel
import tensorflow as tf


class Crop1(CropBaseModel):

    def decoded_funtion(
        self,
        mixed_input,
        mask_source1,
        mask_source2,
        reconstructed_source1,
        reconstructed_source2,
        init_placeholder,
    ):

        # Estimación de la fuente 1
        reconstructed_source1, mask_source1, predictions_1 = (
            self.best_filtered_var_sigmoid(
                reconstructed_source2, mixed_input, self.alpha_2
            )
        )
        self.alpha_2 *= self.beta

        eps = 1e-6
        mask_sum = tf.maximum(mask_source1 + mask_source2, eps)
        m1 = mask_source1 / mask_sum
        m2 = mask_source2 / mask_sum

        # Aplicar máscaras normalizadas para la próxima iteración
        reconstructed_source1 = tf.clip_by_value(2.0 * mixed_input * m1, 0.0, 1.0)

        # Estimación de la fuente 2
        reconstructed_source2, mask_source2, predictions_2 = (
            self.best_filtered_var_sigmoid(
                reconstructed_source1, mixed_input, self.alpha_1
            )
        )
        self.alpha_1 *= self.beta

        eps = 1e-6
        mask_sum = tf.maximum(mask_source1 + mask_source2, eps)
        m1 = mask_source1 / mask_sum
        m2 = mask_source2 / mask_sum

        reconstructed_source2 = tf.clip_by_value(2.0 * mixed_input * m2, 0.0, 1.0)
        return (
            mask_source1,
            mask_source2,
            reconstructed_source1,
            reconstructed_source2,
            predictions_1,
            predictions_2,
        )
