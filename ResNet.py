import tensorflow as tf


class ResNet(tf.keras.Model):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def _residual(self, x, output_channels, use_1x1conv=False, strides=1):
        start = x

        x = tf.keras.layers.Conv2D(
            filters=output_channels, kernel_size=3, padding="same", strides=strides
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.activations.relu(x)
        x = tf.keras.layers.Conv2D(
            filters=output_channels, kernel_size=3, padding="same"
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)

        if use_1x1conv:
            start = tf.keras.layers.Conv2D(
                filters=output_channels, kernel_size=1, strides=strides
            )(start)
        x += start
        x = tf.keras.activations.relu(x)
        return x

    def _ResNet_block(self, x, output_channels, num_residuals, first_block=False):
        for i in range(num_residuals):
            if i == 0 and not first_block:
                # cut each of height and width by half
                x = self._residual(x, output_channels, use_1x1conv=True, strides=2)
            else:
                x = self._residual(x, output_channels)
        return x

    def call(self, x):
        x = tf.keras.layers.Conv2D(
            filters=64, kernel_size=7, strides=2, padding="same"
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.activations.relu(x)
        x = tf.keras.layers.MaxPooling2D(
            pool_size=3, strides=2, padding="same"
        )(x)

        x = self._ResNet_block(x, 64, 2, first_block=True)
        x = self._ResNet_block(x, 128, 2)
        x = self._ResNet_block(x, 256, 2)
        x = self._ResNet_block(x, 512, 2)
        
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(units=self.num_classes, activation="softmax")(x)
        return x

    def build(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape)
        outputs = self.call(inputs)
        return tf.keras.Model(
            inputs=inputs, outputs=outputs,
            name="ResNet-18"
        )

if __name__ == "__main__": 
    model = ResNet(num_classes=1000)
    model = model.build((224, 224, 3))
    model.summary()

