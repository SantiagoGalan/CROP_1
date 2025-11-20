import tensorflow as tf
from project.CROP.models.crop_base_model import CropBaseModel


class Crop1(CropBaseModel):

    def decoded_function(
        self,
        mixed_input,
        mask_source1,
        mask_source2,
        reconstructed_source1,
        reconstructed_source2,
        params,
    ):

        alpha_1 = params["alpha_1"]
        alpha_2 = params["alpha_1"]
        beta = params["beta"]
        # Estimación de la fuente 1
        reconstructed_source1, mask_source1, predictions_1 = (
            self.best_filtered_var_sigmoid(reconstructed_source2, mixed_input, alpha_2)
        )
        alpha_2 *= beta

        # Estimación de la fuente 2
        reconstructed_source2, mask_source2, predictions_2 = (
            self.best_filtered_var_sigmoid(reconstructed_source1, mixed_input, alpha_1)
        )
        alpha_1 *= beta

        return (
            mask_source1,
            mask_source2,
            reconstructed_source1,
            reconstructed_source2,
            predictions_1,
            predictions_2,
        )
