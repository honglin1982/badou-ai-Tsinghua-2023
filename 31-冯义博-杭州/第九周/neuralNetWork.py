import numpy as np
import scipy.special



class neuralNetWork:

    def __init__(self, inputNum, hiddenNum, outputNum, learnRate):
        self.input = inputNum
        self.hidden = hiddenNum
        self.output = outputNum
        self.lr = learnRate
        #  初初始化权重

        self.wih = np.random.normal(0, pow(self.hidden, -0.5), (self.hidden, self.input))
        self.who = np.random.normal(0, pow(self.output, -0.5), (self.output, self.hidden))

        # 激活函数使用sigmoid
        self.activation_function = lambda x: scipy.special.expit(x)

    """
    推理
    """

    def query(self, inputs):
        # 计算输入层--》隐藏层 （输入*权重）
        aih = np.dot(self.wih, inputs)
        # 过激活函数
        aih = self.activation_function(aih)
        # 计算隐藏层--》输出层
        aho = np.dot(self.who, aih)
        aho = self.activation_function(aho)
        # print("aho:", aho)
        return aho

    """
    训练
    """

    def train(self, input_list, target_list):
        # 输入数组转换为2维数组 ndmin指定维度
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T
        # 计算隐藏节点输出 输入节点*权重
        aih = np.dot(self.wih, inputs)
        aih = self.activation_function(aih)
        # 计算输出节点输出
        aho = np.dot(self.who, aih)
        aho = self.activation_function(aho)
        # 计算误差 简单求差值 可以套公式用MSE 交叉熵
        output_errors = targets - aho
        hidden_errors = np.dot(self.who.T, output_errors * aho * (1 - aho))
        # 反向传播 套公式
        self.who += self.lr * np.dot((output_errors * aho * (1 - aho)), np.transpose(aih))
        self.wih += self.lr * np.dot((hidden_errors * aih * (1 - aih)), np.transpose(inputs))
        pass


if __name__ == "__main__":
    network = neuralNetWork(28 * 28, 512, 10, 0.5)
    # 读取训练数据
    training_data_file = open("dataset/mnist_train.csv")
    training_data_list = training_data_file.readlines()
    training_data_file.close()
    # epoch为20
    for e in range(20):
        for record in training_data_list:
            all_values = record.split(',')
            inputs = (np.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
            # 设置图片与数值的对应关系
            targets = np.zeros(10) + 0.01
            targets[int(all_values[0])] = 0.99
            network.train(inputs, targets)

    # 读取测试集
    test_data_file = open("dataset/mnist_test.csv")
    test_data_list = test_data_file.readlines()
    test_data_file.close()
    scores = []
    for record in test_data_list:
        all_values = record.split(',')
        correct_number = int(all_values[0])
        print("该图片对应的数字为:", correct_number)
        # 处理数据
        inputs = (np.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
        # 让网络判断图片对应的数字
        outputs = network.query(inputs)
        # 找到数值最大的神经元对应的编号
        label = np.argmax(outputs)
        print("网络认为图片的数字是：", label)
        if label == correct_number:
            scores.append(1)
        else:
            scores.append(0)
    print(scores)

    # 计算图片判断的成功率
    scores_array = np.asarray(scores)
    print("perfermance = ", scores_array.sum() / scores_array.size)
