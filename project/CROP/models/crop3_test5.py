"""CROP3 sin corrección local de saturación."""

import tensorflow as tf

from project.CROP.models.crop3_test3 import Crop3Test3


class Crop3Test5(Crop3Test3):
    """Omite los promedios de vecinos y solo recorta intensidades a [0, 1].

    Mantiene delta, gamma y la proyección de mezcla de Crop3Test3.
    object_threshold y window no intervienen en esta operación.
    """

    @staticmethod
    def correct_local_saturation(values, object_threshold=0.05, window=5):
        return tf.clip_by_value(tf.convert_to_tensor(values), 0, 1)
