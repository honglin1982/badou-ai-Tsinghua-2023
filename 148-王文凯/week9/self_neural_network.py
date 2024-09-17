from keras.datasets import mnist
from keras.utils import to_categorical
import scipy.special
import numpy as np

class self_neural_network:
    # 初始化函数
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate, epoch):
        self.ins = input_nodes
        self.hns = hidden_nodes
        self.ons = output_nodes
        self.lr = learning_rate
        self.epoch = epoch

        # 权重矩阵初始化
        # input -> hidden
        self.wih = np.random.rand(self.ins, self.hns) - 0.5
        # hidden -> output
        self.who = np.random.rand(self.hns, self.ons) - 0.5
        # 激活函数 sigmoid
        self.sigmoid = lambda x: scipy.special.expit(x)
        # 激活函数 relu
        self.relu = lambda x: np.maximum(0, x)

    # softmax
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    # 模型训练函数
    def train(self, data_list, data_labels):
        for _ in range(self.epoch):
            for data_item, label_item in zip(data_list, data_labels):
                data = np.expand_dims(data_item, axis=0)
                # print('data_shape', data.shape)
                label = np.expand_dims(label_item, axis=0)
                # print('labels_shape', labels.shape)
                # input -> hidden
                hidden_inputs = np.dot(data, self.wih)
                # print('hidden_inputs_shape', hidden_inputs.shape)
                # hidden_input -> hidden_output
                hidden_outputs = self.sigmoid(x=hidden_inputs)
                # print('hidden_outputs_shape', hidden_outputs.shape)
                # hidden -> output
                final_inputs = np.dot(hidden_outputs, self.who)
                # print('final_inputs_shape', final_inputs.shape)
                final_outputs = self.sigmoid(x=final_inputs)
                # print('final_outputs_shape', final_outputs.shape)
                output_errors = label - final_outputs
                # print('output_errors_shape', output_errors.shape)
                hidden_errors = np.dot(self.who, output_errors.T * final_outputs.T * (1 - final_outputs.T))
                # print('hidden_errors_shape', hidden_errors.shape)
                self.who += self.lr * np.dot(np.transpose(hidden_outputs),
                                             output_errors * final_outputs * (1 - final_outputs))
                self.wih += self.lr * np.dot(np.transpose(data),
                                             hidden_errors.T * hidden_outputs * (1 - hidden_outputs))

    # 模型推理函数
    def query(self, cur_item):
        # input -> hidden
        # print('input:', cur_item)
        hidden_input = np.dot(cur_item, self.wih)
        # print('hidden_input:', hidden_input)
        # hidden_input -> hidden_output
        hidden_output = self.sigmoid(x=hidden_input)
        # print('hidden_output:', hidden_output)
        # hidden -> output
        final_input = np.dot(hidden_output, self.who)
        # print('final_input:', final_input)
        final_output = self.sigmoid(x=final_input)
        # print('final_output:', final_output)
        return self.softmax(final_output)

def main():
    # 数据预处理
    (train_data_list, train_labels), (test_data_list, test_labels) = mnist.load_data()
    train_data_list = train_data_list.reshape(60000, 28 * 28)
    train_data_list = train_data_list.astype('float') / 255
    test_data_list = test_data_list.reshape(10000, 28 * 28)
    test_data_list = test_data_list.astype('float') / 255
    train_labels = to_categorical(train_labels)

    network = self_neural_network(input_nodes=784, hidden_nodes=512, output_nodes=10, learning_rate=0.3, epoch=5)
    network.train(train_data_list, train_labels)
    scores = []
    for (cur_data, correct_number) in zip(test_data_list, test_labels):
        predict_number = np.argmax(network.query(cur_data))
        scores.append(1 if predict_number == correct_number else 0)
        print('正确数字:', correct_number)
        print('网络判断结果:', predict_number)
    scores_array = np.asarray(scores)
    print("正确率 = ", scores_array.sum() / scores_array.size)


if __name__ == '__main__':
    main()
