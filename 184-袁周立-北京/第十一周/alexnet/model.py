from keras.models import Sequential
from keras import layers
import tensorflow as tf


def AlexNet_Model(output_shape=1000):

    model = Sequential([

        # layers.Input(shape=(224, 224, 3)),

        # # 以下方式是不是不建议
        # layers.Lambda(lambda image: tf.image.resize(image, (227, 227)) / 255),

        layers.Conv2D(96, 11, strides=(4, 4), padding='valid', activation="relu", input_shape=(227, 227, 3)),
        # layers.Conv2D(96, 11, strides=(4, 4), padding='valid', activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(3, strides=(2, 2), padding="valid"),

        layers.Conv2D(256, 5, padding='same', activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(3, strides=(2, 2), padding="valid"),

        layers.Conv2D(384, 3, padding='same', activation="relu"),
        layers.Conv2D(384, 3, padding='same', activation="relu"),
        layers.Conv2D(256, 3, padding='same', activation="relu"),
        layers.MaxPooling2D(3, strides=(2, 2), padding="valid"),
        layers.Flatten(),

        layers.Dense(4096, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(4096, activation="relu"),
        layers.Dropout(0.3),

        layers.Dense(output_shape, activation="softmax")
    ])

    # model.build(input_shape=input_shape)
    model.summary()

    return model


if __name__ == "__main__":
    AlexNet_Model(output_shape=100)
