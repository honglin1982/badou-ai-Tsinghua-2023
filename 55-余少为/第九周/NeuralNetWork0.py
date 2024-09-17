import numpy as np
import scipy.special


class NeuralNetWork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # 初始化网络，设置输入层、中间层和输出层节点数
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.lr = learningrate  # 学习率

        # 设置权重矩阵，初始化为-0.5到0.5之间。
        # w：权重 i：输入层 h：中间层 o：输出层
        self.wih = np.random.rand(self.hnodes, self.inodes) - 0.5
        self.who = np.random.rand(self.onodes, self.hnodes) - 0.5

        # 设定激活函数
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    def train(self, inputs_list, targets_list):
        # 根据输入的训练数据，更新节点链路权重
        # 把输入和输出转换为len*1的二维数组
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # 正向计算
        hidden_inputs = np.dot(self.wih, inputs)  # 计算中间层
        hidden_outputs = self.activation_function(hidden_inputs)  # 过激活函数
        final_inputs = np.dot(self.who, hidden_outputs)  # 计算输出层
        final_outputs = self.activation_function(final_inputs)  # 过激活函数

        # 计算误差,反向传播并更新权重
        outputs_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, outputs_errors*final_outputs*(1-final_outputs))
        self.who += self.lr * np.dot(outputs_errors*final_outputs*(1-final_outputs), np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot(hidden_errors*hidden_outputs*(1-hidden_outputs), np.transpose(inputs))

        pass

    def query(self, inputs):
        # 根据输入数据，计算并给出预测结果
        hidden_inputs = np.dot(self.wih, inputs)    # 计算中间层
        hidden_outputs = self.activation_function(hidden_inputs)     # 过激活函数
        final_inputs = np.dot(self.who, hidden_outputs)     # 计算输出层
        final_outputs = self.activation_function(final_inputs)     # 过激活函数

        return final_outputs


input_nodes = 784       # 输入图片是28*28=784
hidden_nodes = 256      # 设置200个中间节点
output_nodes = 10       # 输出10个分类，代表数字0-9
learning_rate = 0.3     # 设置学习率
epochs = 5      # 训练轮数
n = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)   # 初始化神经网络

# 读入训练数据
training_data_file = open('dataset/mnist_train.csv', 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# 开始训练
for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(',')
        inputs = np.asfarray(all_values[1:]) / 255.0 * 0.99 + 0.01
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)

# 读入测试集，评估模型正确率
test_data_file = open('dataset/mnist_test.csv')
test_data_list = test_data_file.readlines()
test_data_file.close()
scores = []
for record in test_data_list:
    all_values = record.split(',')
    correct_number = int(all_values[0])
    print('该图片对应的数字为:', correct_number)
    inputs = np.asfarray(all_values[1:]) / 255.0 * 0.99 + 0.01
    outputs = n.query(inputs)
    label = np.argmax(outputs)
    print('网络预测图片数字为：', label)
    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)
print(scores)

scores_array = np.asarray(scores)
print('perfermance = ', scores_array.sum() / scores_array.size)

