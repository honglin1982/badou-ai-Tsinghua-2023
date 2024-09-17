from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.datasets import mnist


def AlexNet(input_shape=(224, 224, 3), output_shape=2):
    model = Sequential()
    model.add(
        Conv2D(
            filters=48,
            kernel_size=(11, 11),
            strides=(4, 4),
            padding='valid',
            input_shape=input_shape,
            activation='relu'
        )
    )
    model.add(BatchNormalization())

    model.add(
        MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding='valid'
        )
    )

    model.add(
        Conv2D(
            filters=128,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )

    model.add(BatchNormalization())
    # 使用步长为2的最大池化层进行池化，此时输出的shape为(13,13,256)；
    model.add(
        MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding='valid'
        )
    )
    # 使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的特征层为384层，输出的shape为(13,13,384)；
    # 所建模型后输出为192特征层
    model.add(
        Conv2D(
            filters=192,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )
    # 使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的特征层为384层，输出的shape为(13,13,384)；
    # 所建模型后输出为192特征层
    model.add(
        Conv2D(
            filters=192,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )
    # 使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的特征层为256层，输出的shape为(13,13,256)；
    # 所建模型后输出为128特征层
    model.add(
        Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )
    # 使用步长为2的最大池化层进行池化，此时输出的shape为(6,6,256)；
    model.add(
        MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding='valid'
        )
    )

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(output_shape, activation='softmax'))

    return model
