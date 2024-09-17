# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    """
    tf.placeholder( dtype, shape=None, name=None)
    dtype：数据类型。常用的是tf.float32,tf.float64等数值类型
    shape：数据形状。默认是None，就是一维值，也可以是多维（比如[2,3], [None, 3]表示列是3，行不定）
    name：名称
    """
    x = tf.placeholder(tf.float32, [None, 1], 'x')
    y = tf.placeholder(tf.float32, [None, 1], 'y')

    # 定义神经网络中间层
    weight_l1 = tf.Variable(tf.random_normal([1, 10]), name='w1')
    biases_l1 = tf.Variable(tf.zeros([10]), name='b1')
    wx_plus_b_l1 = tf.matmul(x, weight_l1, name='X') + biases_l1
    # 激活函数
    l1 = tf.nn.tanh(wx_plus_b_l1,name='softmax')

    # 定义神经网络输出层
    weight_l2 = tf.Variable(tf.random_normal([10, 1]), name='w2')
    biases_l2 = tf.Variable(tf.zeros([1]), name='b2')
    wx_plus_b_l2 = tf.matmul(l1, weight_l2, name='X') + biases_l2
    # 激活函数
    out = tf.nn.tanh(wx_plus_b_l2, name='softmax')

    # 定义损失函数
    loss = tf.reduce_mean(tf.square(y - out), name='loss')
    # 定义反向传播算法（梯度下降算法）
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    init = tf.global_variables_initializer()
    with tf.Session() as session:
        # 使用numpy生成200个随机点
        x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
        y_data = np.square(x_data) + np.random.normal(0, 0.02, x_data.shape)

        session.run(init)
        epochs = 20000
        for epoch in range(epochs):
            session.run(train_step, feed_dict={x: x_data, y: y_data})

        result = session.run(out, feed_dict={x: x_data})

        # 画图
        plt.figure()
        plt.scatter(x_data, y_data)  # 散点是真实值
        plt.plot(x_data, result, 'r-', lw=5)  # 曲线是预测值
        plt.show()
