from project.CROP.models.crop_base_model import CropBaseModel
from project.custom_layers.sampling import Sampling
import tensorflow as tf



class Crop3MaskCorrection(CropBaseModel):


    def normalize_image(self, x, eps=1e-8):

        x = tf.convert_to_tensor(x)

        # detectar dimensiones automáticamente
        ndims = len(x.shape)

        if ndims == 4:
            axes = [1, 2, 3]

        elif ndims == 3:
            axes = [1, 2]

        elif ndims == 2:
            axes = [0, 1]

        else:
            return x

        x_min = tf.reduce_min(x, axis=axes, keepdims=True)
        x_max = tf.reduce_max(x, axis=axes, keepdims=True)

        x_norm = (x - x_min) / (x_max - x_min + eps)

        return tf.clip_by_value(x_norm, 0, 1)

    def _show_images(self, tensor, title="", max_images=1):
        if not hasattr(self, "debug") or not self.debug:
            return

        imgs = tensor

        if isinstance(imgs, tf.Tensor):
            imgs = imgs.numpy()

        # asegurar batch
        if len(imgs.shape) == 1:
            imgs = np.expand_dims(imgs, axis=0)

        n = min(max_images, imgs.shape[0])

        fig, axes = plt.subplots(1, n, figsize=(n * 2, 2))

        if n == 1:
            axes = [axes]

        for i in range(n):
            img = imgs[i]

            # reshape si está plano
            if len(img.shape) == 1:
                img = img.reshape(28, 28)

            max_val = np.max(img)
            min_val = np.min(img)
            axes[i].imshow(img, cmap="gray")
            axes[i].set_title(f"{title} \nmax: {max_val:.3f}, min: {min_val:3f} ", fontsize=8)
            #axes[i].set_title(f"{title}", fontsize=8)
            axes[i].axis("off")

        plt.tight_layout()
        plt.show()

    def filter(self, filter_1 ,mixed_input, alpha, bias, slope):

        x_mix_filter_1 =  2 * mixed_input - filter_1
        
        x_mix_filter_1 = tf.clip_by_value(x_mix_filter_1, 0, 1)

        condition_encoder = self.predictor(x_mix_filter_1, verbose=0, training=False)
        encoded_imgs = self.cvae.encoder(
            [x_mix_filter_1, condition_encoder], verbose=0, training=0
        )
        
        zz_log_var = encoded_imgs[1] + alpha

        z = Sampling()((encoded_imgs[0], zz_log_var))

        cvae_output = self.cvae.decoder(
            [z, condition_encoder], verbose=0, training=False
        )
        
        mask_source1 = (cvae_output - bias) * slope

        mask_source1 = tf.sigmoid(mask_source1)
    

        x_mix_filter_1 = 2 * mixed_input * mask_source1

        x_mix_filter_1 = tf.clip_by_value(x_mix_filter_1, 0, 1)

        return x_mix_filter_1, mask_source1, condition_encoder

    def second_filter(self, filter_1 ,mixed_input, alpha, bias, slope,delta):
        #print("second filter")
        x_mix_filter_1 =  2 * mixed_input - filter_1
        
        x_mix_filter_1 = tf.clip_by_value(x_mix_filter_1, 0, 1)

        condition_encoder = self.predictor(x_mix_filter_1, verbose=0, training=False)
        encoded_imgs = self.cvae.encoder(
            [x_mix_filter_1, condition_encoder], verbose=0, training=0
        )
        
        zz_log_var = encoded_imgs[1] + alpha

        z = Sampling()((encoded_imgs[0], zz_log_var))

        cvae_output = self.cvae.decoder(
            [z, condition_encoder], verbose=0, Training=False
        )
        
        mask_source1 = (cvae_output - bias) * slope

        mask_source1 = tf.sigmoid(mask_source1)
        
        self._show_images(mask_source1,"mask generada")
        
        mask_source1 = mask_source1 - delta*((self.source1_estimation+self.source2_estimation)/2-mixed_input)
        

        self._show_images(mask_source1, "mascara post delta ")
        
        
        self._show_images(mixed_input,"mezlca orignal")
        
        x_mix_filter_1 = 2 * mixed_input * mask_source1

        x_mix_filter_1 = tf.clip_by_value(x_mix_filter_1, 0, 1)

        self._show_images(x_mix_filter_1, "resultado de un filtrado")
            


        return x_mix_filter_1, mask_source1, condition_encoder



    def decode(self):
        alpha_1 = self.model_params["alpha_1"]
        alpha_2 = self.model_params["alpha_2"]
        beta = self.model_params["beta"]
        bias = self.model_params["bias"]
        slope = self.model_params["slope"]
        gamma = self.model_params["gamma"]
        delta = self.model_params["delta"]
        iteration = self.iteration
        threshold = self.model_params["threshold"]

        if iteration<threshold: # puede ser otra valor de iter.
            reconstructed_source1, mask_source1, predictions_1 = (
                self.filter(self.source2_estimation ,self.mixed_input, alpha_2, bias, slope)
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
                self.filter(reconstructed_source1,  self.mixed_input, alpha_1, bias, slope)
            )

            self.source2_estimation = reconstructed_source2
            self.mask2 = mask_source2
            self.predictions2 = predictions_2
            self.model_params["alpha_1"] = alpha_1 * beta

            x__x = (self.source1_estimation + self.source2_estimation) / 2
            x__x_e = x__x - self.mixed_input
            
            self.source2_estimation = self.source2_estimation - (x__x_e * gamma)

            self.source2_estimation = tf.clip_by_value(self.source2_estimation, 0, 1)
        else:
            
            #print("-----------------------Segundo Ciclo---------------")
            self._show_images(self.mask1,"mask 1")
            self._show_images(self.mask2, "mask 2")
            self._show_images(self.source1_estimation,"fuente estimada 1 inicio de iteracion")
            self._show_images(self.source2_estimation, "fuente estimada 2 inicio de iteracion")
            

            x__x = (self.source1_estimation+self.source2_estimation)/2
            mix_error = x__x - self.mixed_input
            self._show_images(mix_error, "error entre mezclas")

            reconstructed_source1, mask_source1, predictions_1 = (
                self.second_filter(self.source2_estimation ,self.mixed_input, alpha_2, bias, slope,delta)
            )

            self.source1_estimation = reconstructed_source1
            self.mask1 = mask_source1
            self.predictions1 = predictions_1
            self.model_params["alpha_2"] = alpha_2 * beta

            x__x = (self.source1_estimation + self.source2_estimation) / 2
            x__x_e = x__x - self.mixed_input
            

            self.source1_estimation = self.source1_estimation - (x__x_e * gamma)

       
            self.source1_estimation = tf.clip_by_value(self.source1_estimation, 0, 1)



            self._show_images(self.source1_estimation,"fuente estimada 1 final de iteracion")
            

            reconstructed_source2, mask_source2, predictions_2 = (
                self.second_filter(reconstructed_source1,  self.mixed_input, alpha_1, bias, slope,delta)
            )

            self.source2_estimation = reconstructed_source2
            self.mask2 = mask_source2
            self.predictions2 = predictions_2
            self.model_params["alpha_1"] = alpha_1 * beta

            x__x = (self.source1_estimation + self.source2_estimation) / 2
            x__x_e = x__x - self.mixed_input
            
            self.source2_estimation = self.source2_estimation - (x__x_e * gamma)

            self.source2_estimation = tf.clip_by_value(self.source2_estimation, 0, 1)


            self._show_images(self.source2_estimation, "fuente estimada 2 final de iteracion")
