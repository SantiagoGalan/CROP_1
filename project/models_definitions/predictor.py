import tensorflow as tf
from keras.layers import Dense, Conv2D, BatchNormalization, Dropout, Flatten
from project.custom_layers.reshapeLayer import ReshapeLayer


class Predictor(tf.keras.Model):
    def __init__(
        self,
        image_size=28,
        n_cond=10,
        dropout_rate=0.5,
        name="predictor",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.image_size = image_size
        self.original_dim = image_size * image_size
        self.original_dim_C = (image_size, image_size, 1)

        # Layers
        self.reshape = ReshapeLayer(self.original_dim_C)

        self.conv1 = Conv2D(
            32, 3, strides=2, padding="same", activation="relu"
        )
        self.bn1 = BatchNormalization()

        self.conv2 = Conv2D(
            64, 3, strides=2, padding="same", activation="relu"
        )
        self.bn2 = BatchNormalization()

        self.flatten = Flatten()
        self.dropout = Dropout(dropout_rate)

        self.classifier = Dense(n_cond, activation="softmax")

    def call(self, inputs, training=False):
        x = self.reshape(inputs)

        x = self.conv1(x)
        x = self.bn1(x, training=training)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        x = self.flatten(x)
        x = self.dropout(x, training=training)

        return self.classifier(x)
