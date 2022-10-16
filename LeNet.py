import tensorflow as tf

class LeNet(tf.keras.Model):
    def __init__(self, original=True):
        super().__init__()
        self.original = original

    def _feature_extraction(self, x):
        if self.original:
            nonlinearity = "sigmoid"
            pooling = tf.keras.layers.AveragePooling2D
        else:
            nonlinearity = "relu"
            pooling = tf.keras.layers.MaxPooling2D
        
        x = tf.keras.layers.Conv2D(
            filters=6, kernel_size=(5, 5), activation=nonlinearity
        )(x)
        x = pooling(pool_size=(2, 2))(x)
        x = tf.keras.layers.Conv2D(
            filters=16, kernel_size=(5, 5), activation=nonlinearity
        )(x)
        x = pooling(pool_size=(2, 2))(x)
        x = tf.keras.layers.Flatten()(x)
        return x

    def _classifier(self, x):
        nonlinearity = "sigmoid" if self.original else "relu"
        x = tf.keras.layers.Dense(units=120, activation=nonlinearity)(x)
        x = tf.keras.layers.Dense(units=84, activation=nonlinearity)(x)
        x = tf.keras.layers.Dense(units=10, activation="softmax", name="softmax")(x)
        return x

    def build(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape)
        outputs = self.call(inputs)
        return tf.keras.Model(
            inputs=inputs, outputs=outputs,
            name="LeNet"
        )

    def call(self, x):
        x = self._feature_extraction(x)
        x = self._classifier(x)
        return x


if __name__ == "__main__": 
    model = LeNet()
    model = model.build((28, 28, 1))
    model.summary()

