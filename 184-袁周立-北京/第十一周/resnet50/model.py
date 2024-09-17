import keras
from keras import layers
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions


def ResNet50_Model():
    inputs = keras.Input(shape=(224, 224, 3))

    # stage0
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(3, strides=(2, 2), padding="same")(x)   # --> (56, 56, 64)

    # stage1
    x = ConvBlock(x, kernel_sizes=[1, 3, 1], filters=[64, 64, 256], strides=(1, 1))
    x = IdentityBlock(x, kernel_sizes=[1, 3, 1], filters=[64, 64, 256])
    x = IdentityBlock(x, kernel_sizes=[1, 3, 1], filters=[64, 64, 256])

    # stage2
    x = ConvBlock(x, kernel_sizes=[1, 3, 1], filters=[128, 128, 512], strides=(2, 2))
    x = IdentityBlock(x, kernel_sizes=[1, 3, 1], filters=[128, 128, 512])
    x = IdentityBlock(x, kernel_sizes=[1, 3, 1], filters=[128, 128, 512])
    x = IdentityBlock(x, kernel_sizes=[1, 3, 1], filters=[128, 128, 512])

    # stage3
    x = ConvBlock(x, kernel_sizes=[1, 3, 1], filters=[256, 256, 1024], strides=(2, 2))
    x = IdentityBlock(x, kernel_sizes=[1, 3, 1], filters=[256, 256, 1024])
    x = IdentityBlock(x, kernel_sizes=[1, 3, 1], filters=[256, 256, 1024])
    x = IdentityBlock(x, kernel_sizes=[1, 3, 1], filters=[256, 256, 1024])
    x = IdentityBlock(x, kernel_sizes=[1, 3, 1], filters=[256, 256, 1024])
    x = IdentityBlock(x, kernel_sizes=[1, 3, 1], filters=[256, 256, 1024])

    # stage4
    x = ConvBlock(x, kernel_sizes=[1, 3, 1], filters=[512, 512, 2048], strides=(2, 2))
    x = IdentityBlock(x, kernel_sizes=[1, 3, 1], filters=[512, 512, 2048])
    x = IdentityBlock(x, kernel_sizes=[1, 3, 1], filters=[512, 512, 2048])

    x = layers.AveragePooling2D((7, 7))(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(1000, activation="softmax")(x)

    model = keras.Model(inputs, outputs=outputs, name="ResNet50_Model")
    model.summary()
    return model


def ConvBlock(inputs, kernel_sizes, filters, strides):
    return Block(inputs, kernel_sizes, filters, strides, conv_block=True)


def IdentityBlock(inputs, kernel_sizes, filters):
    return Block(inputs, kernel_sizes, filters, strides=(1, 1), conv_block=False)


def Block(inputs, kernel_sizes, filters, strides, conv_block=True):
    kernel_size1, kernel_size2, kernel_size3 = kernel_sizes
    filter1, filter2, filter3 = filters

    # stage1不会减少网格大小，因为stage1前面有max_pool减少过网格大小了
    x = layers.Conv2D(filter1, kernel_size1, strides=strides, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filter2, kernel_size2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filter3, kernel_size3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    short_cut = inputs
    if conv_block:
        short_cut = layers.Conv2D(filter3, (1, 1), strides=strides, padding="same")(inputs)
        short_cut = layers.BatchNormalization()(short_cut)

    x = layers.add([x, short_cut])
    x = layers.Activation("relu")(x)

    return x


if __name__ == "__main__":
    resnet50_model = ResNet50_Model()
    resnet50_model.load_weights("./test/resnet50_weights_tf_dim_ordering_tf_kernels.h5")

    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    img_elephant = cv2.imread("./test/elephant.jpg")
    img_bike = cv2.imread("./test/bike.jpg")

    img_elephant = cv2.cvtColor(img_elephant, cv2.COLOR_BGR2RGB)
    img_elephant = cv2.resize(img_elephant, (224, 224))
    img_bike = cv2.cvtColor(img_bike, cv2.COLOR_BGR2RGB)
    img_bike = cv2.resize(img_bike, (224, 224))

    # 无须归一化等处理，测试结果也是正确的
    x = np.concatenate(
        (np.expand_dims(img_elephant, axis=0),  np.expand_dims(img_bike, axis=0)),
        axis=0
    )

    preds = resnet50_model.predict(x)
    decode_pred = decode_predictions(preds)
    print('elephant Predicted:', decode_pred[0])
    print('bike Predicted:', decode_pred[1])


