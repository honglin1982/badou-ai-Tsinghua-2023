from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization


class AlexNetModel:
    """
    这里为了加快收敛速度，卷积层 filters / 2，
    全连接层减为1024
    output_shape: 这里是分类猫和狗，所以为 2
    """

    def __init__(self, input_shape: tuple, output_shape=2):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = Sequential()

    def conv2d_1(self):
        """ 卷积1 """
        # 使用步长为4x4，大小为11的卷积核对图像进行卷积，输出的特征层为96层，输出的shape为(55,55,96)；
        # 所建模型后输出为48特征层
        self.model.add(
            Conv2D(
                filters=48,
                kernel_size=(11, 11),
                strides=(4, 4),
                padding="valid",
                input_shape=self.input_shape,
                activation='relu'
            )
        )

    def max_pooling1(self):
        """ 池化1 """
        # 使用步长为2的最大池化层进行池化，此时输出的shape为(27,27,96)
        self.model.add(
            MaxPooling2D(
                pool_size=(3, 3),
                strides=(2, 2),
                padding="valid",
            )
        )

    def conv2d_2(self):
        """ 卷积2 """
        # 使用步长为1x1，大小为5的卷积核对图像进行卷积，输出的特征层为256层，输出的shape为(27,27,256)；
        # 所建模型后输出为128特征层
        self.model.add(
            Conv2D(
                filters=128,
                kernel_size=(5, 5),
                strides=(1, 1),
                padding="same",
                activation='relu'
            )
        )

    def max_pooling2(self):
        """ 池化2 """
        # 使用步长为2的最大池化层进行池化，此时输出的shape为(13,13,256)；
        self.model.add(
            MaxPooling2D(
                pool_size=(3, 3),
                strides=(2, 2),
                padding="valid",
            )
        )

    def conv2d_3(self):
        """ 卷积3 """
        # 使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的特征层为384层，输出的shape为(13,13,384)；
        # 所建模型后输出为192特征层
        self.model.add(
            Conv2D(
                filters=192,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="same",
                activation='relu'
            )
        )

    def conv2d_4(self):
        """ 卷积4 """
        # 使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的特征层为384层，输出的shape为(13,13,384)；
        # 所建模型后输出为192特征层
        self.model.add(
            Conv2D(
                filters=192,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="same",
                activation='relu'
            )
        )

    def conv2d_5(self):
        """ 卷积5 """
        # 使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的特征层为256层，输出的shape为(13,13,256)；
        # 所建模型后输出为128特征层
        self.model.add(
            Conv2D(
                filters=128,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="same",
                activation='relu'
            )
        )

    def max_pooling3(self):
        """ 池化3 """
        # 使用步长为2的最大池化层进行池化，此时输出的shape为(6,6,256)；
        self.model.add(
            MaxPooling2D(
                pool_size=(3, 3),
                strides=(2, 2),
                padding="valid",
            )
        )

    def forward(self):
        """ 全连接2层 """
        # 两个全连接层，最后输出为1000类,这里改为2类
        # 缩减为1024
        self.model.add(Flatten())  # 拍扁，降维
        self.model.add(Dense(units=1024, activation='relu'))
        self.model.add(Dropout(0.25))

        self.model.add(Dense(units=1024, activation='relu'))
        self.model.add(Dropout(0.25))

        self.model.add(Dense(units=self.output_shape, activation='softmax'))

    def produce_model(self) -> Sequential:
        self.conv2d_1()
        self.model.add(BatchNormalization())
        self.max_pooling1()
        self.conv2d_2()
        self.model.add(BatchNormalization())
        self.max_pooling2()
        self.conv2d_3()
        self.conv2d_4()
        self.conv2d_5()
        self.max_pooling3()
        self.forward()
        return self.model
