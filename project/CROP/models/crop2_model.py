from CROP_models.crop_base_model import CropBaseModel

# from  crop_base_model import CropBaseModel
import tensorflow as tf


class Crop2(CropBaseModel):

    def decoded_funtion(
        self,
        mixed_input,
        mask_source1,
        mask_source2,
        reconstructed_source1,
        reconstructed_source2,
        init_placeholder,
    ):
        reconstructed_source1, mask_source1, predictions_1 = (
            self.best_filtered_var_sigmoid(
                reconstructed_source2, mixed_input, self.alpha_2
            )
        )

        self.alpha_2 = self.alpha_2 * self.beta

        #
        x__x = (reconstructed_source1 + reconstructed_source2) / 2

        x__x_e = x__x - mixed_input

        reconstructed_source1 = reconstructed_source1 - (x__x_e * self.gamma)

        reconstructed_source1 = tf.clip_by_value(
            reconstructed_source1, clip_value_min=0, clip_value_max=1
        )

        reconstructed_source2, mask_source2, predictions_2 = (
            self.best_filtered_var_sigmoid(
                reconstructed_source1, mixed_input, self.alpha_1
            )
        )

        self.alpha_1 = self.alpha_1 * self.beta

        x__x = (reconstructed_source1 + reconstructed_source2) / 2
        x__x_e = x__x - mixed_input

        reconstructed_source2 = reconstructed_source2 - (x__x_e * self.gamma)

        reconstructed_source2 = tf.clip_by_value(
            reconstructed_source2, clip_value_min=0, clip_value_max=1
        )

        return (
            mask_source1,
            mask_source2,
            reconstructed_source1,
            reconstructed_source2,
            predictions_1,
            predictions_2,
        )
