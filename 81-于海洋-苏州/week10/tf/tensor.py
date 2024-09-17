#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@date 2023/7/12
@author: 81-yuhaiyang

"""
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


def gen_data(count):
    print("gen_data")
    # [200, 1]  的二维数组
    x = np.linspace(-0.5, 0.5, count)[:, np.newaxis]
    noise = np.random.normal(0, 0.02, x.shape)
    y = np.square(x) + noise
    return x, y


def gen_hide_layer(input_x, input_node: int, out_node: int):
    # 目前仅仅设计了一层隐藏层
    print("gen hide layer")
    w = tf.Variable(tf.random_normal([input_node, out_node]))
    b = tf.Variable(tf.zeros([input_node, out_node]))
    # y = wx + b
    y = tf.matmul(input_x, w) + b
    res = tf.nn.tanh(y)  # 激活函数
    return res


def gen_out_layer(input_x, real_y, in_node: int):
    print("gen out layer")
    w = tf.Variable(tf.random_normal([in_node, 1]))
    b = tf.Variable(tf.zeros([1, 1]))
    y = tf.matmul(input_x, w) + b

    loss = tf.reduce_mean(tf.square(real_y - y))
    back = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    return y, back


if __name__ == '__main__':
    data_x, data_y = gen_data(1000)

    # 相当于输入层
    var_x = tf.placeholder(tf.float32, [None, 1])
    var_y = tf.placeholder(tf.float32, [None, 1])

    hide_l_1 = gen_hide_layer(var_x, 1, 20)
    out_y, step = gen_out_layer(hide_l_1, var_y, 20)

    with tf.Session() as s:
        s.run(tf.global_variables_initializer())

        for i in range(2000):
            s.run(step, feed_dict={var_x: data_x, var_y: data_y})

        predict_y = s.run(out_y, feed_dict={var_x: data_x})

        # 画图
        plt.figure()
        plt.scatter(data_x, data_y)  # 散点是真实值
        plt.plot(data_x, predict_y, 'r-', lw=5)  # 曲线是预测值
        plt.show()
