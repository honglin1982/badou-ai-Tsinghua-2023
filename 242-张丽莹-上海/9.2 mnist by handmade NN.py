# 手动创建神经网络以识别手写数字

import numpy
import scipy.special

# 类：神经网络
class NeuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # 初始化网络，设置输入层，中间层，和输出层节点数
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # 学习率
        self.lr = learningrate

        # 随机初始化权重矩阵.
        # 法1 创建均匀分布在-0.5-0.5间的权重值。使用random.rand函数
        # self.wih = numpy.random.rand(self.hnodes, self.inodes) - 0.5 # 输入和隐层之间的w
        # self.who = numpy.random.rand(self.onodes, self.hnodes) - 0.5 # 隐层和输出曾之间的w

        # 法2 创建正态分布均值设置为0.0，标准差设置为pow(self.hnodes,-0.5)：self.hnodes的-0.5次方
        # 这样生成的随机数会更加接近于0，并且标准差的倒数作为分布的尺度参数，有助于保持梯度的稳定性。
        self.wih = (numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes)))
        self.who = (numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes)))

        # 激活函数是sigmoid。经过激活函数的结果将作为下一次输入
        self.activation_function = lambda x:scipy.special.expit(x)

        pass

    # 训练函数：更新权重的。参数：输入数据、标签值
    def train(self, inputs_list, targets_list):
        # 先将输入和标签做成矩阵，注意矩阵乘法的shape
        inputs = numpy.array(inputs_list, ndmin=2).T  #ndmin是指定维度
        targets = numpy.array(targets_list, ndmin=2).T

        # h1 = 输入 x w1
        hidden_inputs = numpy.dot(self.wih, inputs)
        # h1经激活函数变成下一层输入
        hidden_outputs = self.activation_function(hidden_inputs)
        # 最后一层（就这么几层是因为这个代码在简单演示）
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # 计算误差
        output_errors = targets - final_outputs  # 注意这俩矩阵格式要相同
        hidden_errors = numpy.dot(self.who.T, output_errors * final_outputs * (1 - final_outputs))
        # 根据误差计算链路权重的更新量，然后把更新加到原来链路权重上。依据是链式法则公式
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1 - final_outputs)),
                                        numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                        numpy.transpose(inputs))

        pass

    # 正向推理函数：根据输入数据计算并输出答案
    def query(self,inputs):
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        print(final_outputs)
        return final_outputs

# 主函数
# 1.初始化网络
input_nodes = 784   # 每张图有28x28=784个数值
hidden_nodes = 200  # 中间节点数量，自定
output_nodes = 10   # 结果是十分类
learning_rate = 0.1 # 学习率，自定
# 代入做好的网络结构，init函数的参数
n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# 2.读入训练数据,open函数-readlines函数-close函数
training_data_file = open("dataset/minist.csv", 'r')
training_data_list = training_data_file.readlines()  #读取文件的所有行，并将其存储为一个列表。每一行数据都是文件中的一条训练数据。
training_data_file.close()  # close函数释放资源

# 3.设定epochs、对目标标签进行编码
epochs = 5
for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(',') # 每行数据用,隔开
        inputs = (numpy.asfarray(all_values[1:]))/255.0 * 0.99 + 0.01
        # asfarray是转为浮点数，all_values[1:]是除了第一个元素，其他从0-255转为0-1再转为0.01-1
        # 多类别分类问题中的目标标签编码：
        targets = numpy.zeros(output_nodes) + 0.01
        # 先设定和目标输出一样大小的矩阵，初始都设成0.01
        targets[int(all_values[0])] = 0.99
        # 对应目标类别的元素值为0.99
        n.train(inputs, targets)
        # 实例训练

# 4.测试集载入
test_data_file = open("dataset/mnist_test.csv")
test_data_list = test_data_file.readlines()
test_data_file.close()

# 测试集进行正向推理
scores = []
for record in test_data_list:
    all_values = record.split(',')
    correct_number = int(all_values[0])
    print("该图片对应的数字为:",correct_number)  # 记录标签正确值

    # 预处理数字图片
    inputs = (numpy.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
    # 让网络判断图片对应的数字
    outputs = n.query(inputs)
    # 输出数值最大的神经元的索引，在这里也就是数字数值
    label = numpy.argmax(outputs)  # argmax就是输出最大值的索引
    print("网络认为图片的数字是：", label)

    # 统计得分
    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)
print(scores)
print("perfermance = ", sum(scores) / len(scores))

