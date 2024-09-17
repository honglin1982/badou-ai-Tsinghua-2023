from keras import layers
from keras.models import Sequential
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
import tensorflow as tf
import numpy as np


def VGG16_Model():
    model = Sequential([

        layers.Conv2D(64, 3, padding="same", activation="relu", input_shape=(224, 224, 3), name="conv1_1"),
        layers.Conv2D(64, 3, padding="same", activation="relu", name="conv1_2"),
        layers.MaxPooling2D(2, name="pool1"),

        layers.Conv2D(128, 3, padding="same", activation="relu", name="conv2_1"),
        layers.Conv2D(128, 3, padding="same", activation="relu", name="conv2_2"),
        layers.MaxPooling2D(2, name="pool2"),

        layers.Conv2D(256, 3, padding="same", activation="relu", name="conv3_1"),
        layers.Conv2D(256, 3, padding="same", activation="relu", name="conv3_2"),
        layers.Conv2D(256, 3, padding="same", activation="relu", name="conv3_3"),
        layers.MaxPooling2D(2, name="pool3"),

        layers.Conv2D(512, 3, padding="same", activation="relu", name="conv4_1"),
        layers.Conv2D(512, 3, padding="same", activation="relu", name="conv4_2"),
        layers.Conv2D(512, 3, padding="same", activation="relu", name="conv4_3"),
        layers.MaxPooling2D(2, name="pool4"),

        layers.Conv2D(512, 3, padding="same", activation="relu", name="conv5_1"),
        layers.Conv2D(512, 3, padding="same", activation="relu", name="conv5_2"),
        layers.Conv2D(512, 3, padding="same", activation="relu", name="conv5_3"),
        layers.MaxPooling2D(2, name="pool5"),

        # 下载的h5权重中，是先flatten再接上全连接层的，所以不能用1*1卷积模拟
        # layers.Conv2D(4096, 7, padding="valid", activation="relu", name="fc6"),
        # layers.Conv2D(4096, 1, padding="valid", activation="relu", name="fc7"),
        # layers.Conv2D(1000, 1, padding="valid", activation="softmax", name="fc8"),
        # layers.Flatten()

        layers.Flatten(),
        layers.Dense(4096, activation='relu', name='fc1'),
        layers.Dense(4096, activation='relu', name='fc2'),
        layers.Dense(1000, activation='softmax', name='fc3')

    ], name="vgg_16")

    model.summary()

    return model


if __name__ == "__main__":
    # VGG16_Model()

    img_dog = image.load_img("./test/dog.jpg", target_size=(224, 224))
    img_table = image.load_img("./test/table.jpg", target_size=(224, 224))

    img_dog = np.expand_dims(img_dog, axis=0)
    img_table = np.expand_dims(img_table, axis=0)

    x = np.concatenate((img_dog, img_table), axis=0)
    x = preprocess_input(x)

    model = VGG16_Model()

    model.load_weights('./test/vgg16_weights_tf_dim_ordering_tf_kernels.h5')

    preds = model.predict(x)
    preds = decode_predictions(preds)

    print("dog pred:", preds[0])
    print("table pred:", preds[1])
