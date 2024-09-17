import numpy as np
import scipy.special


class NeuralNetwork:
    def __init__(self, input_node, hidden_node, output_node, learning_rate):
        # 初始化: 输入层节点、隐藏层节点、输出层节点和学习率
        self.input_node = input_node
        self.hidden_node = hidden_node
        self.output_node = output_node
        self.learn_rate = learning_rate
        """
        初始化权重矩阵：
            1. input_weight_matrix: 表示输入层和中间层节点间链路权重形成的矩阵
            2. output_weight_matrix: 表示中间层和输出层间链路权重形成的矩阵
        """
        self.input_weight_matrix = np.random.normal(0.0, pow(self.hidden_node, -0.5), (self.hidden_node, self.input_node))
        self.output_weight_matrix = np.random.normal(0.0, pow(self.output_node, -0.5), (self.output_node, self.hidden_node))
        # 初始化激活函数: sigmoid(x) = 1 / (1 + exp(-x)); 得到的结果将作为信号输出到下一层
        self.activation_function = lambda x: scipy.special.expit(x)  # 这里的activation_function相当于sigmoid函数

    def train(self, input_list, right_result):
        # 1 数据转换: 最小维度2, 并做转置
        input_data = np.array(input_list, ndmin=2).T
        right_result = np.array(right_result, ndmin=2).T
        # 2 计算信号经过输入层后产生的信号量
        hidden_input = np.dot(self.input_weight_matrix, input_data)
        # 3 中间层神经元对输入的信号做激活函数后得到的输出信号
        hidden_output = self.activation_function(hidden_input)
        # 4 输出层接收来自中间层的输出信号
        final_input = np.dot(self.output_weight_matrix, hidden_output)
        # 5 输出层对接收的信号做激活函数后得到最终的输出信号
        final_output = self.activation_function(final_input)

        # 6 计算误差
        output_error = right_result - final_output
        hidden_error = np.dot(self.output_weight_matrix.T, output_error*final_output*(1-final_output))
        # 7 权重更新：根据误差计算链路权重的更新量，然后把更新加到原来链路权重上
        self.input_weight_matrix += self.learn_rate * np.dot(hidden_error*hidden_output*(1-hidden_output), np.transpose(input_data))
        self.output_weight_matrix += self.learn_rate * np.dot(output_error*final_output*(1-final_output), np.transpose(hidden_output))

        loss = np.mean(np.square(output_error))  # 均方误差计算
        accuracy = np.mean(np.equal(np.argmax(final_output, axis=0), np.argmax(right_result, axis=0)))  # 预测正确的样本数除以总样本数

        return loss, accuracy

    def forecast(self, input_data):
        # 根据输入数据预测结果
        hidden_input = np.dot(self.input_weight_matrix, input_data)
        hidden_output = self.activation_function(hidden_input)
        final_input = np.dot(self.output_weight_matrix, hidden_output)
        final_output = self.activation_function(final_input)
        return final_output


# 2 训练数据
def train_data(network, epoch):
    # 1 读取训练数据
    with open("./dataset/mnist_train.csv", "r") as file:
        data = file.readlines()
    # 2 根据 epoch数 做数据完整训练的循环次数
    for e in range(epoch):
        total_loss = 0.0
        total_accuracy = 0.0
        for t in data:  # 遍历每一条数据
            t2 = t.strip().split(",")  # 数据格式处理
            # 将输入数据转换为浮点型的 NumPy 数组，并做归一化处理
            input_data = np.asfarray(t2[1:]) / 255
            # 将正确结果做独热编码转换
            right_result = np.zeros(output_node)
            right_result[int(t2[0])] = 1
            # 训练数据: 传入训练数据及正确结果
            loss, accuracy = network.train(input_data, right_result)
            total_loss += loss
            total_accuracy += accuracy

        avg_loss = total_loss / len(data)
        avg_accuracy = total_accuracy / len(data)
        print(f"Epoch {e + 1}: Average Loss = {avg_loss:.4f}, Average Accuracy = {avg_accuracy:.4f}")


# 3 推理数据
def inference(network):
    scores = []  # 用于记录推理结果是否成功
    with open("./dataset/mnist_test.csv", "r") as f:
        data = f.readlines()
    for t in data:
        t = t.strip().split(",")
        right_result = int(t[0])  # 正确结果
        print(f"图像正确结果: {right_result}")
        input_data = np.asfarray(t[1:]) / 255  # 输入数据处理
        output_data = network.forecast(input_data)  # 推理数据
        # 找到数值最大的神经元对应的编号
        label = np.argmax(output_data)
        print(f"图像推理结果: {label}\n")
        if label == right_result:
            scores.append(1)
        else:
            scores.append(0)
    # 计算成功率
    print("Inference success rate: ", sum(scores)/len(scores))


if __name__ == '__main__':
    input_node = 784  # 输入节点数：根据数据格式决定
    hidden_node = 200  # 中间层节点数
    output_node = 10  # 输出节点数：根据正确结果的表现形式决定
    learning_rate = 0.1  # 学习率
    nw = NeuralNetwork(input_node, hidden_node, output_node, learning_rate)
    train_data(nw, 10)  # 训练数据，epoch=5
    inference(nw)  # 推理数据

