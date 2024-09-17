import keras
from keras import layers


def CiFar10_Model(input_shape):
    inputs = keras.Input(input_shape)
    x = layers.Conv2D(32, 5, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(200, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(100, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    model = keras.Model(inputs, outputs=outputs, name="cifar10_model")
    model.summary()
    return model


if __name__ == "__main__":
    CiFar10_Model((32, 32, 3))
