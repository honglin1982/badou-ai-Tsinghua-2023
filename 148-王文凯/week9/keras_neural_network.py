from tensorflow import keras

# 通过keras实现一个简单的神经网络 用于识别手写数字
# keras中文文档 https://keras.io/zh/getting-started/sequential-model-guide/

# 调用数据集出错 解决方式 手动下载数据集mnist
# https://github.com/MLEveryday/100-Days-Of-ML-Code/blob/master/datasets/mnist.npz

def main():
    # keras.datasets.mnist 一个机器学习领域的经典数据集 28 * 28 的图像 0-9的分类
    # 训练集 train_data_list 训练集中每项对应正确结果 train_data_labels
    # 测试集 test_data_list 测试集中每项对应正确结果 test_data_labels
    (train_data_list, train_data_labels), (test_data_list, test_data_labels) = keras.datasets.mnist.load_data()

    # step_1 数据初始化

    # 将数据集 从 60000 个 28 * 28 的二维数组 转变为 60000 个 28 * 28 的一维数组
    # 将数据归一化 将原本0-255区间的灰度值 转变为浮点值再转变至0-1区间
    train_data_list = train_data_list.reshape(60000, (28 * 28))
    test_data_list = test_data_list.reshape(10000, (28 * 28))
    train_data_list = train_data_list.astype('float32') / 255
    test_data_list = test_data_list.astype('float32') / 255

    # 通过独热编码处理图像的分类特征
    # keras.utils.to_categorical 用于将 类向量 转换成 二进制类矩阵
    train_data_labels = keras.utils.to_categorical(train_data_labels)
    test_data_labels = keras.utils.to_categorical(test_data_labels)

    # step_2 搭建神经网络
    # keras Sequential 顺序模型 多个网络层的顺序堆叠 单独堆叠方法 model.add
    network = keras.models.Sequential([
        # 隐藏层 512层 激活函数为relu input_shape 一个表示尺寸的元组
        keras.layers.Dense(512, input_shape=(28 * 28, )),
        keras.layers.Activation('relu'),
        # 输出层 10层 激活函数为softmax 用于将 输出结果 映射到0-1区间 理解为 模型识别结果的概率
        keras.layers.Dense(10),
        keras.layers.Activation('softmax'),
    ])

    # 模型编译
    # keras.model.compile 编译函数 三个参数 optimizer 优化器 loss 损失函数 metrics 评估标准
    # 对于任何分类问题 都希望将metrics 设置为 ['accuracy']
    network.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])

    # step_3 模型训练
    # fit 模型训练函数 14个参数 x 训练数据 y 数据正确标记 epochs 训练多少代 batch_size 一次训练多少个数据 ...
    network.fit(x=train_data_list, y=train_data_labels, epochs=10, batch_size=128)

    # step_4 模型测试
    # evaluate 模型评估函数 6个参数
    # x 测试集
    # y 测试集标签 batch_size 每次评估样本数 默认为2
    # verbose 日志显示模式 0 安静模式 1 进度条
    # sample_weight 可选numpy权重数组 用于对损失函数的加权
    # steps 声明评估结束之前的总步数（批次样本）默认值 None
    test_loss, test_acc = network.evaluate(x=test_data_list, y=test_data_labels, verbose=1)
    print('loss:', test_loss)
    print('accuracy:', test_acc)

if __name__ == '__main__':
    main()
