from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras import models
import cifa_data as cd
import tensorflow as tf


class CifarModel:

    def model(self):
        model = models.Sequential()
        # 构建第1卷积层 ci=3 co=63
        input_shape = (24, 24, 3)
        model.add(Conv2D(
            filters=64,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding='same',
            input_shape=input_shape,
            activation='relu'
        ))

        # 池化
        model.add(MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding='same'
        ))

        # 第二个卷积层 ci=64 co=64
        model.add(Conv2D(
            filters=64,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding='same',
            activation='relu'
        ))

        # 池化
        model.add(MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding='same'
        ))

        # 全连接层 拍扁
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(10, activation='softmax'))
        return model


if __name__ == "__main__":
    # data_dir = "cifar_data/cifar-10-batches-bin"
    (x_train, y_train), (x_test, y_test) = cd.inputs()
    network = CifarModel()
    model = network.model()
    # 训练
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, batch_size=100)
    # 测试 verbose: 0 或 1。日志显示模式。 0 = 安静模式，1 = 进度条。
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print("test_loss:", test_loss)
    print("test_acc:", test_acc)

