from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Conv2D,MaxPooling2D,Flatten,Dropout,BatchNormalization
# from tensorflow.keras.datasets import mnist
# from tensorflow.keras.optimizers import Adam


# 定义 AlexNet 神经网络模型:
def AlexNet(input_shape=(224,224,3), output_shape=2):
    model = Sequential()  # 创建一个空的顺序模型
    # 1: 二维卷积层,使用步长为4x4，大小为11的卷积核对图像进行卷积，输出的特征层为96层，输出的shape为(55,55,96)；
    model.add(
        Conv2D(
            filters=48,  # 卷积核数量48,进行了减半操作减少计算量
            kernel_size=(11,11),  # 卷积核大小
            strides=(4,4),  # 步长 4x4
            padding="valid",  # 为有效填充，用于减小特征图的大小，以减少模型参数和计算量
            input_shape=input_shape,
            activation="relu"  # 激活函数
        )
    )
    model.add(BatchNormalization())  # 规范化输入数据,每个卷积层和全连接层之后使用
    # 2: 二维最大池化层:步长为2的池化，此时输出的shape为(27,27,96)
    model.add(
        MaxPooling2D(
            pool_size=(3,3),
            strides=(2,2),
            padding="valid"
        )
    )

    # 3: 二维卷积层: 步长 1x1，大小为5x5的卷积核对图像进行卷积，输出的特征层为256层，输出的shape为(27,27,256)
    model.add(
        Conv2D(
            filters=128,
            kernel_size=(5,5),
            strides=(1,1),
            padding="same",
            activation="relu"
        )
    )
    model.add(BatchNormalization())  # 规范化输入数据
    # 4: 二维最大池化层:步长为2的池化，此时输出的shape为(13,13,256)
    model.add(
        MaxPooling2D(
            pool_size=(3,3),
            strides=(2,2),
            padding="valid"
        )
    )

    # 5: 二维卷积层: 步长 1x1，大小为3x3的卷积核对图像进行卷积，输出的特征层为384层，输出的shape为(13,13,384)
    model.add(
        Conv2D(
            filters=192,
            kernel_size=(3,3),
            strides=(1,1),
            padding="same",
            activation="relu"
        )
    )
    model.add(
        Conv2D(
            filters=192,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation="relu"
        )
    )

    # 6: 二维卷积层: 步长 1x1，大小为3x3的卷积核对图像进行卷积，输出的特征层为256层，输出的shape为(13,13,256)
    model.add(
        Conv2D(
            filters=128,
            kernel_size=(3,3),
            strides=(1,1),
            padding="same",
            activation="relu"
        )
    )
    # 7: 二维最大池化层:步长为2的池化，此时输出的shape为(6,6,256)
    model.add(
        MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding="valid"
        )
    )
    # 8.展平层
    model.add(Flatten())
    # 9.全连接层,神经元数量1024
    model.add(Dense(units=1024, activation="relu"))
    # 该层防止过拟合：每次训练迭代中随机丢弃 25% 的神经元的输出
    model.add(Dropout(0.25))
    # 10.全连接层
    model.add(Dense(units=1024,activation="relu"))
    model.add(Dropout(0.25))
    # 输出层，输出通道2
    model.add(Dense(units=output_shape,activation="softmax"))

    return model
