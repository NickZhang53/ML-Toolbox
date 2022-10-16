import tensorflow as tf

class VGGNet(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def _feature_extraction(self, x):
        def conv(filters):
            return tf.keras.layers.Conv2D(
                filters=filters, 
                kernel_size=(3, 3), 
                strides=1, 
                padding="same", 
                activation="relu"
            )
        pool = tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2), strides=2
        )

        # Input: 224 * 224 * 3
        x = conv(64)(x)
        x = conv(64)(x)
        # 224 * 224 * 64
        x = pool(x)
        # 112 * 112 * 64
        x = conv(128)(x)
        x = conv(128)(x)
        # 112 * 112 * 128
        x = pool(x)
        # 56 * 56 * 128
        x = conv(256)(x)
        x = conv(256)(x)
        x = conv(256)(x)
        # 56 * 56 * 256
        x = pool(x)
        # 28 * 28 * 256
        x = conv(512)(x)
        x = conv(512)(x)
        x = conv(512)(x)
        # 28 * 28 * 512
        x = pool(x)
        # 14 * 14 * 512
        x = conv(512)(x)
        x = conv(512)(x)
        x = conv(512)(x)
        # 14 * 14 * 512
        x = pool(x)
        # 7 * 7 * 512
        x = tf.keras.layers.Flatten()(x)
        return x

    def _classifier(self, x):
        x = tf.keras.layers.Dense(
            units=4096, activation="relu"
        )(x)
        x = tf.keras.layers.Dense(
            units=4096, activation="relu"
        )(x)
        x = tf.keras.layers.Dense(
            units=1000, activation="softmax"
        )(x)
        return x

    def build(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape)
        outputs = self.call(inputs)
        return tf.keras.Model(
            inputs=inputs, outputs=outputs,
            name="VGG16"
        )

    def call(self, x):
        x = self._feature_extraction(x)
        x = self._classifier(x)
        return x

if __name__ == "__main__": 
    model = VGGNet()
    model = model.build((224, 224, 3))
    model.summary()


