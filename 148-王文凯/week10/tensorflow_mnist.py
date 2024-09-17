import tensorflow as tf
from tensorflow.keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np

# 定义一个可以实现手写数字识别的简单神经网络生成类 返回一个基于tf的神经网络模型
class mnist_simple_neural_network:

    # 初始化函数
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rates, epoch):
        self.ins = input_nodes
        self.hns = hidden_nodes
        self.ons = output_nodes
        self.lr = learning_rates
        self.epoch = epoch

        self.x = tf.compat.v1.placeholder(tf.float32, [None,  self.ins])
        self.y = tf.compat.v1.placeholder(tf.float32, [None, self.ons])

        # without hidden
        # self.w = tf.Variable(tf.zeros([self.ins, self.ons]))
        # self.b = tf.Variable(tf.zeros([self.ons]))
        # self.output_res = tf.nn.softmax(tf.matmul(self.x, self.w) + self.b)

        # with hidden
        self.w_hidden_input = tf.Variable(tf.random.normal([self.ins, self.hns]))
        self.b_hidden_input = tf.Variable(tf.random.normal([self.hns]))
        self.hidden_input = tf.matmul(self.x, self.w_hidden_input) + self.b_hidden_input
        self.hidden_output = tf.nn.softmax(self.hidden_input)

        self.w_output_input = tf.Variable(tf.random.normal([self.hns, self.ons]))
        self.b_output_input = tf.Variable(tf.random.normal([self.ons]))
        self.output_input = tf.matmul(self.hidden_output, self.w_output_input) + self.b_output_input
        self.output_res = tf.nn.softmax(self.output_input)

        # with conv
        # self.w_conv1 = self.weight_variable([5, 5, 1, 32])
        # self.b_conv1 = self.bias_variable([32])
        # self.h_conv1 = tf.nn.relu(self.conv2d(self.x, self.w_conv1) + self.b_conv1)
        # self.h_pool1 = self.max_pool_2x2(self.h_conv1)
        #
        # self.w_conv2 = self.weight_variable([5, 5, 32, 64])
        # self.b_conv2 = self.bias_variable([64])
        # self.h_conv2 = tf.nn.relu(self.conv2d(self.h_pool1, self.w_conv2) + self.b_conv2)
        # self.h_pool2 = self.max_pool_2x2(self.h_conv2)
        #
        # self.w_fc1 = self.weight_variable([7 * 7 * 64, 1024])
        # self.b_fc1 = self.bias_variable([1024])
        #
        # self.h_pool12_flat = tf.reshape(self.h_pool2, [-1, 7 * 7 * 64])
        # self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool12_flat, self.w_fc1) + self.b_fc1)
        #
        # self.keep_prob = tf.placeholder(tf.float32)
        # self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)
        #
        # self.w_fc2 = self.weight_variable([1024, 10])
        # self.b_fc2 = self.bias_variable([10])
        #
        # self.output_res = tf.nn.softmax(tf.matmul(self.h_fc1_drop, self.w_fc2) + self.b_fc2)

        # 损失函数 交叉熵
        self.loss = -tf.reduce_sum(self.y * tf.log(self.output_res + 1e-10))

        # 定义反向传播算法
        self.train_step = tf.compat.v1.train.GradientDescentOptimizer(self.lr).minimize(self.loss)
        # ADAM 优化器
        # self.train_step = tf.compat.v1.train.AdamOptimizer(1e-4).minimize(self.loss)

        # 初始化
        self.sess = tf.compat.v1.Session()
        self.init = tf.compat.v1.global_variables_initializer()
        self.sess.run(self.init)

    # 训练函数
    def train(self, train_data_list, train_data_labels):
        for _ in range(self.epoch):
            for i in range(0, len(train_data_list)):
                batch_img = np.expand_dims(train_data_list[i], 0)
                batch_label = np.expand_dims(train_data_labels[i], 0)
                self.sess.run(self.train_step, feed_dict={self.x: batch_img, self.y: batch_label})

    # 推理函数
    def query(self, test_data):
        return self.sess.run(self.output_res, feed_dict={self.x: np.expand_dims(test_data, 0)})

    # 卷积
    # def weight_variable(self, shape):
    #     initial = tf.truncated_normal(shape, stddev=0.1)
    #     return tf.Variable(initial)
    #
    # def bias_variable(self, shape):
    #     initial = tf.constant(0.1, shape=shape)
    #     return tf.Variable(initial)
    #
    # def conv2d(self, x, W):
    #     return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    #
    # def max_pool_2x2(self, x):
    #     return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
    #                           strides=[1, 2, 2, 1], padding='SAME')


def main():
    # 数据预处理
    (train_data_list, train_data_labels), (test_data_list, test_data_labels) = mnist.load_data()
    train_data_list = train_data_list.reshape(60000, 784)
    test_data_list = test_data_list.reshape(10000, 784)
    train_data_list = train_data_list.astype('float') / 255
    test_data_list = test_data_list.astype('float') / 255
    train_data_labels = to_categorical(train_data_labels)

    network = mnist_simple_neural_network(input_nodes=784, hidden_nodes=512, output_nodes=10, learning_rates=0.1, epoch=10)
    network.train(train_data_list, train_data_labels)
    scores = []
    for (cur_data, correct_number) in zip(test_data_list, test_data_labels):
        predict_number = np.argmax(network.query(cur_data))
        scores.append(1 if predict_number == correct_number else 0)
        print('正确数字:', correct_number)
        print('网络判断结果:', predict_number)
    scores_array = np.asarray(scores)
    print("正确率 = ", scores_array.sum() / scores_array.size)

if __name__ == '__main__':
    main()
