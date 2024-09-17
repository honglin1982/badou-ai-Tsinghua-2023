#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@date 2023/7/12
@author: 81-yuhaiyang

"""
import tensorflow as tf


def x_plus_b():
    var_x = tf.placeholder(tf.int32)
    var_b = tf.placeholder(tf.int32)

    op_x = tf.multiply(var_x, var_x)
    op_y = tf.add(op_x, var_b)

    tip = "Feed 方式 测试：y = x^2 + b 请输入x值:"
    input_x = int(input(tip))
    tip = "请输入b值:"
    input_b = int(input(tip))

    with tf.Session() as sess:
        res = sess.run([op_y, op_x], feed_dict={var_x: input_x, var_b: input_b})
        print("op_y:", res[0])
        print("op_x:", res[1])
        sess.close()


def x_plus_b_2():
    var_x = tf.Variable(0, name="x")
    var_b = tf.Variable(0, name="b")

    tip = "Variable 方式 测试：y = x^2 + b 请输入x值:"
    input_x = int(input(tip))
    tip = "请输入b值:"
    input_b = int(input(tip))

    var_x = tf.assign(var_x, input_x)
    var_b = tf.assign(var_b, input_b)

    init_op = tf.global_variables_initializer()

    op_x = tf.multiply(var_x, var_x)
    op_y = tf.add(op_x, var_b)

    with tf.Session() as sess:
        res = sess.run(init_op)
        res = sess.run([op_y, op_x])
        print("x*x = ", res[1])
        print("x*x+b = ", res[0])
        sess.close()


if __name__ == '__main__':
    x_plus_b_2()
