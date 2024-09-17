# 导入 mnist 数据集,MNIST 数据集是一个广泛使用的手写数字图像数据集
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras import models  # 用于构建、训练和评估深度学习模型的类和函数
from tensorflow.keras import layers  # 用于构建深度学习模型的不同层
from tensorflow.keras.utils import to_categorical  # 该函数用于将整数型的类别标签转换为独热编码形式
import numpy as np


# 输入参数: 完整训练代数
def keras_neural_network(epoch):
    # 1 加载数据集:第一次运行需要下载数据，会比较慢
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    print(f"训练集形状: {train_images.shape}")  # 每一张数字图像的大小是 28*28
    print(f"训练集标签形状: {train_labels.shape}")  # 标签对应的值表示数字图像对应的数字
    print(f"测试集形状: {test_images.shape}")
    print(f"测试集标签形状: {test_labels.shape}")
    # 测试打印如下：
    # plt.imshow(train_images[0], cmap=plt.cm.binary)
    # plt.title(f"number:{train_labels[0]}")
    # plt.show()

    # 2 使用tensorflow.keras 搭建一个有效识别图案的神经网络
    neural_network = models.Sequential()  # 创建一个空的顺序模型对象
    # 添加一个具有512个神经元和 ReLU 激活函数的全连接层(layers.Dense)到顺序模型 neural_network 中，并指定输入的形状为 (28*28,)
    neural_network.add(layers.Dense(512, activation="relu", input_shape=(28*28,)))
    # 添加一个具有10个神经元和softmax激活函数的全连接层到顺序模型 neural_network 中。这一层将作为模型的输出层，用于分类任务中的多类别分类
    neural_network.add(layers.Dense(10, activation="softmax"))
    # 编译模型的语句，用于配置模型的优化器、损失函数和评估指标
    neural_network.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

    # 3 将图像数组格式从3维转为2维，并归一化处理
    train_images = train_images.reshape((60000, 28*28)).astype("float32") / 255
    test_images = test_images.reshape((10000, 28*28)).astype("float32") / 255

    # 4 独热编码转换:用于将整数型的类别标签转换为独热编码（one-hot encoding）的形式
    print(f"before labels: {train_labels[0]}")
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    print("after labels: ", train_labels[0])

    # 5 训练：将数据输入网络进行训练,epochs=5 进行5次完整训练，batch_size=128 每次训练分为多批训练：60000/128
    neural_network.fit(train_images, train_labels, epochs=epoch, batch_size=128)

    # 6 推理：测试数据输入，检验训练学习后的图片识别效果（识别效果与硬件相关）
    # 使用给定的测试数据集对模型进行评估，并返回评估结果，如测试损失和测试准确率
    test_loss, test_accuracy = neural_network.evaluate(test_images, test_labels, verbose=3)
    print(f"测试集损失值: {test_loss}；测试集准确率: {test_accuracy}")

    # 7 预测: 输入一张手写数字图片到网络中，测试识别效果
    while True:
        index = input("请输入你要预测的测试集数据索引 (00 退出):")
        if index == "00":
            break
        index = int(index)
        print(f"预测数据的正确结果: {test_labels[index].argmax()}")
        predictions = neural_network.predict(np.array([test_images[index]]))
        np.set_printoptions(suppress=True, precision=3)  # 设置打印选项，禁止使用科学计数法，并设置小数精度
        print(test_labels[index])
        print(predictions[0])
        print(f"预测结果: {predictions[0].argmax()}, 对应概率为:{predictions[0][predictions[0].argmax()]}")


if __name__ == '__main__':
    keras_neural_network(5)

