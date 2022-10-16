import tensorflow as tf

class AlexNet(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def _feature_extraction(self, x):
        x = tf.keras.layers.Conv2D(
            filters=96, kernel_size=(11, 11), strides=4, activation="relu"
        )(x)
        x = tf.keras.layers.MaxPooling2D(
            pool_size=(3, 3), strides=2
        )(x)
        x = tf.keras.layers.Conv2D(
            filters=256, kernel_size=(5, 5), padding="same", activation="relu"
        )(x)
        x = tf.keras.layers.MaxPooling2D(
            pool_size=(3, 3), strides=2
        )(x)
        x = tf.keras.layers.Conv2D(
            filters=384, kernel_size=(3, 3), padding="same", activation="relu"
        )(x)
        x = tf.keras.layers.Conv2D(
            filters=384, kernel_size=(3, 3), padding="same", activation="relu"
        )(x)
        x = tf.keras.layers.Conv2D(
            filters=256, kernel_size=(3, 3), padding="same", activation="relu"
        )(x)
        x = tf.keras.layers.MaxPooling2D(
            pool_size=(3, 3), strides=2
        )(x)
        x = tf.keras.layers.Flatten()(x)
        return x

    def _classifier(self, x):
        x = tf.keras.layers.Dense(
            units=4096, activation="relu"
        )(x)
        x = tf.keras.layers.Dropout(rate=0.5)(x)
        x = tf.keras.layers.Dense(
            units=4096, activation="relu"
        )(x)
        x = tf.keras.layers.Dropout(rate=0.5)(x)
        x = tf.keras.layers.Dense(
            units=1000, activation="softmax"
        )(x)
        return x

    def build(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape)
        outputs = self.call(inputs)
        return tf.keras.Model(
            inputs=inputs, outputs=outputs,
            name="AlexNet"
        )

    def call(self, x):
        x = self._feature_extraction(x)
        x = self._classifier(x)
        return x


if __name__ == "__main__": 
    model = AlexNet()
    model = model.build((227, 227, 3))
    model.summary()

