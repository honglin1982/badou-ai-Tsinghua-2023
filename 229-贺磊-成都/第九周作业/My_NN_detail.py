# -*- coding: utf-8 -*-
# File  : My_NN_detail.py
# Author: HeLei
# Date  : 2023/7/14

import scipy.special
import numpy as np


class My_NeuralNet:

    # 初始化
    def __init__(self, inputnodes, hidedennodes, outputnodes, learningate):
        """
        书写传统神经网络，给3层，输入层，隐藏层(中间层),输出层
        :param inputnodes: 输入层
        :param hidedennodes: 中间层
        :param outputnodes: 输出层
        :param learningate: 学习率
        """
        self.inodes = inputnodes
        self.hnodes = hidedennodes
        self.onodes = outputnodes

        # 设置学习率
        self.lr = learningate
        '''
        初始化权重矩阵，我们有两个权重矩阵，一个是wih表示输入层和中间层节点间链路权重形成的矩阵
        一个是who,表示中间层和输出层间链路权重形成的矩阵
        '''
        # self.woh是一个二维数组，大小为(self.onodes,  self.hnodes)，
        # 其中self.onodes是输出层节点的数量，self.hnodes是隐藏层节点的数量。
        # np.random.rand(self.onodes,self.hnodes)是numpy库的函数，用于生成一个形状为(self.onodes,  self.hnodes)的随机数组。
        # 这个随机数组的取值范围是[0,1)之间。
        # 然后，这个随机数组会减去0.5，即减去0.5的元素-wise操作。所以最终得到的self.woh的元素取值范围是[-0.5,  0.5)之间，也就是在对应的权重范围内的随机初始值。
        self.wih = np.random.rand(self.hnodes, self.inodes) - 0.5
        self.who = np.random.rand(self.onodes, self.hnodes) - 0.5

        # 设置激活函数
        # 这个lambda函数的定义是`lambda  x  :  scipy.special.expit(x)`，参数是x，返回值是`scipy.special.expit(x)`。
        # 对象可以在后续的代码中使用self.activation_function(x)来调用该函数，从而计算激活函数。
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    # 训练
    def train(self, inputs_list, targets_list):
        # 根据输入的训练数据更新节点链路权重
        """
        把inputs_list, targets_list转换成numpy支持的二维矩阵
        .T表示做矩阵的转置
        """
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # 计算信号量经过输入层后产生的信号量
        hidden_inputs = np.dot(self.wih, inputs)

        # 计算中间层经过激活函数后产生的输出信号量
        hidden_outputs = self.activation_function(hidden_inputs)

        # 输出层接受来自中间层的信号量
        final_inputs = np.dot(self.who, hidden_outputs)

        # 输出层做激活函数后得到的最终输出的信号量
        final_outputs = self.activation_function(final_inputs)

        # 以上几行就是正向传播

        # 下面是反向传播

        # 计算损失(误差)
        output_errors = targets - final_outputs

        # output_errors = 1 / 2 * (targets - final_outputs) ** 2 #均方误差效果不太好

        """交叉熵损失函数
        epsilon = 1e-10  # 用于避免log(0)的情况
        final_outputs = np.clip(final_outputs, epsilon, 1.0 - epsilon)
        output_errors = -np.sum(targets * np.log(final_outputs))
        """

        hidden_errors = np.dot(self.who.T, output_errors * final_outputs * (1 - final_outputs))

        # 根据误差计算链路权重地更新量，然后把更新加到原来链路权重上
        self.who += self.lr * np.dot((output_errors * final_outputs * (1 - final_outputs)),
                                     np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                     np.transpose(inputs))

    def inference(self, inputs):
        # 计算中间层从输入层接收到的信号量
        hidden_inputs = np.dot(self.wih, inputs)

        # 计算中间层经过激活函数形成的输出信号量
        hidden_outputs = self.activation_function(hidden_inputs)

        # 计算最后一层接收到的信号量
        final_inputs = np.dot(self.who, hidden_outputs)

        # 计算最后一次层经过激活函数形成的输出信号量
        final_outputs = self.activation_function(final_inputs)

        print("最后的输出层：", final_outputs)

        return final_outputs


# 初始化网络结点个数
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

learning_rate = 0.1
n = My_NeuralNet(input_nodes, hidden_nodes, output_nodes, learning_rate)

# 读数据
train_data = open("NeuralNetWork_从零开始/dataset/mnist_train.csv", "r")
train_data_list = train_data.readlines()  # 读成一个列表
train_data.close()

# 加入epochs
print("开始训练...")
epochs = 5
for epoch in range(epochs):
    # 该csv文件的第一个是标签，后面的是数字，用逗号隔开，
    for record in train_data_list:
        all_value = record.split(",")
        inputs = (np.asfarray(all_value[1:])) / 255.0 * 0.99 + 0.01

        # 设置图片与熟知的对应关系
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_value[0])] = 0.99

        # 开始训练
        n.train(inputs, targets)

# 开始推理
test_data = open("NeuralNetWork_从零开始/dataset/mnist_test.csv", "r")
test_data_list = test_data.readlines()  # 同样读成列表形式
test_data.close()

scores = []  # 用于存放判断正确的数字
for record in test_data_list:
    all_value = record.split(",")
    correct_number = int(all_value[0])  # 标签数据
    print("该图片对应的数字为;", correct_number)

    # 预处理数字图片
    inputs = (np.asfarray(all_value[1:])) / 255.0 * 0.99 + 0.1

    # 推理结果
    result = n.inference(inputs)

    # 找到数值最大的神经元对应的编号
    index = np.argmax(result)
    print("该网络认为的图片数字是:", index)

    if index == correct_number:
        scores.append(1)
    else:
        scores.append(0)
print(scores)

# 计算图片判断的成功率
scores_array = np.asarray(scores)
print("该网络的表现效果:", scores_array.sum() / scores_array.size)
