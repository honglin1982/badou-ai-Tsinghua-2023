from keras.models import Sequential
from keras.layers import Dense, Activation, MaxPooling2D, Flatten, Dropout, BatchNormalization, Conv2D
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import Adam


def AlexNet(input_shape=(224, 224, 3), output_shape=2):
    # 创建AlexNet模块
    model = Sequential()
    # 这里使用步长为4，卷积核大小为11，输出特征层为96，输出形状为（55，55，96）
    # 所建模型输出为48层特征
    model.add(
        Conv2D(
            filters=48,
            kernel_size=(11, 11),
            strides=4,
            padding='valid',
            input_shape=input_shape,

        )
    )

    model.add(BatchNormalization())
    # 使用步长为2的最大池化层进行池化，此时输出为（27，27，96）
    model.add(
        MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding='valid'
        )
    )
    # 使用步长为1，大小为5的卷积核进行卷积，输出结果为（27，27，256）
    # 模型输出结果为128
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
    # 继续池化，步长为2，大小为3，输出结果为（13，13，256）
    model.add(
        MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding='valid'
        )
    )
    # 使用大小为3，步长为1的卷积核进行卷积，输出结果为（13，13，384）
    # 模型输出特征为192层
    model.add(
        Conv2D(
            filters=192,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )
    model.add(
        Conv2D(
            filters=192,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )
    # 使用大小为3，步长为1的卷积核进行卷积，输出结果为（13，13，384）
    # 模型输出特征为192层
    model.add(
        Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )
    # 使用步长为为2，大小为3的最大值池化进行池化，输出形状为（6，6，256）
    model.add(
        MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding='valid'
        )
    )
    # 设置两个全连接层，最后输出为100类，这里改成2类，拍平之后缩减为1024
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(output_shape, activation='softmax'))

    return model
