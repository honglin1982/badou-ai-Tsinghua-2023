import abc
import numpy as np
from abc import ABC
from scipy.special import expit
# import scipy.special


class BaseModel(ABC):

    @abc.abstractmethod
    def train(self, *args, **kwargs):
        raise NotImplemented

    @abc.abstractmethod
    def query(self, *args, **kwargs):
        raise NotImplemented


class NetWorkWrite(BaseModel):
    input_layer = 5
    hidden_layer = 3
    output_layer = 3
    leaning_rate = 0.01

    def __init__(self, *, input_layer=None, hidden_layer=None, output_layer=None, leaning_rate=None):
        self.input_layer = input_layer or self.input_layer
        self.hidden_layer = hidden_layer or self.hidden_layer
        self.output_layer = output_layer or self.output_layer
        self.leaning_rate = leaning_rate or self.leaning_rate
        # 权重值，这里需要注意的是 行列相乘的原则
        '''
        初始化权重矩阵，我们有两个权重矩阵，一个是wih表示输入层和中间层节点间链路权重形成的矩阵
        一个是who,表示中间层和输出层间链路权重形成的矩阵
        '''
        self.wih = np.random.rand(self.hidden_layer, self.input_layer) - 0.5
        self.who = np.random.rand(self.output_layer, self.hidden_layer) - 0.5
        # 激活函数
        '''
        scipy.special.expit对应的是sigmod函数.
        lambda是Python关键字，类似C语言中的宏定义，当我们调用self.activation_function(x)时，编译器会把其转换为spicy.special_expit(x)。
        '''
        self.activate_func = lambda x: expit(x)

    def train(self, input_array, target):
        input_array = np.array(input_array, ndmin=2).T
        target = np.array(target, ndmin=2).T
        # 计算信号经过输入层后产生的信号量
        t_z1 = np.dot(self.wih, input_array)
        # print(t_z1)
        # 中间层神经元对输入的信号做激活函数后得到输出信号
        t_o1 = self.activate_func(t_z1)
        # print(t_o1)
        # 输出层接收来自中间层的信号量
        final_inputs = np.dot(self.who, t_o1)
        # 输出层对信号量进行激活函数后得到最终输出信号
        final_outputs = self.activate_func(final_inputs)
        # 计算误差
        output_errors = target - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors * final_outputs * (1 - final_outputs))
        # 根据误差计算链路权重的更新量，然后把更新加到原来链路权重上
        self.who += self.leaning_rate * np.dot((output_errors * final_outputs * (1 - final_outputs)), np.transpose(t_o1))
        self.wih += self.leaning_rate * np.dot((hidden_errors * t_o1 * (1 - t_o1)), np.transpose(input_array))

    def query(self, input_array: np.ndarray):
        # 根据输入数据计算并输出答案
        # 计算中间层从输入层接收到的信号量
        z1 = np.dot(self.wih, input_array)
        # 计算中间层经过激活函数后形成的输出信号量
        o1 = self.activate_func(z1)
        # 计算最外层接收到的信号量
        final_inputs = np.dot(self.who, o1)
        # 计算最外层神经元经过激活函数后输出的信号量
        final_outputs = self.activate_func(final_inputs)
        print(final_outputs)
        return final_outputs


if __name__ == '__main__':
    input_layer = 784
    hidden_layer = 200
    output_layer = 10
    leaning_rate = 0.1
    net = NetWorkWrite(input_layer=input_layer, hidden_layer=hidden_layer, output_layer=output_layer, leaning_rate=leaning_rate)
    # 读入训练数据
    # open函数里的路径根据数据存储的路径来设定
    training_data_file = open("./dataset/mnist_train.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()
    # 加入epocs,设定网络的训练循环次数
    epochs = 5
    for e in range(epochs):
        # 把数据依靠','区分，并分别读入
        for record in training_data_list:
            all_values = record.split(',')
            inputs = (np.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
            # 设置图片与数值的对应关系
            targets = np.zeros(output_layer) + 0.01
            targets[int(all_values[0])] = 0.99
            net.train(inputs, targets)

    test_data_file = open("dataset/mnist_test.csv")
    test_data_list = test_data_file.readlines()
    test_data_file.close()
    scores = []
    for record in test_data_list:
        all_values = record.split(',')
        correct_number = int(all_values[0])
        print("该图片对应的数字为:", correct_number)
        # 预处理数字图片
        inputs = (np.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
        # 让网络判断图片对应的数字
        outputs = net.query(inputs)
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
