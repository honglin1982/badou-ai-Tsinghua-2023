from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dropout
from tensorflow.keras import models



def model():
    input_shape = (224, 224, 3)
    model = models.Sequential()

    # 第一个卷积层
    model.add(Conv2D(
        filters=48,
        kernel_size=(11, 11),
        strides=(4, 4),
        padding='valid',
        input_shape=input_shape,
        activation='relu'
    ))

    # 归一化处理 使数据落在激活函数敏感区
    model.add(BatchNormalization())

    model.add(MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='valid'
    ))

    # 第二个卷积层
    model.add(Conv2D(
        filters=128,
        strides=(1, 1),
        kernel_size=(5, 5),
        padding='same',
        activation='relu'
    ))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='valid'
    ))

    # 第三个卷积层
    model.add(Conv2D(
        filters=192,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        activation='relu'
    ))

    model.add(Conv2D(
        filters=192,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        activation='relu'
    ))

    model.add(Conv2D(
        filters=128,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        activation='relu'
    ))

    model.add(MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='valid'
    ))

    # 拍扁
    model.add(Flatten())
    # 全连接
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(2, activation='softmax'))
    return model


