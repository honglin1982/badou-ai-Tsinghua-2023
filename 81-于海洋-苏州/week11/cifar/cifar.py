#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@date 2023/7/20
@author: 81-yuhaiyang

"""
import math
import time

import numpy as np

import cifar_10_model
import tensorflow as tf
import cifar_10_data as cifar

if __name__ == '__main__':
    max_steps = 4000
    batch_size = 100
    num_examples_for_eval = 10000
    data_dir = "data/cifar-10-batches-bin"

    images_train, labels_train = cifar.inputs(data_dir=data_dir, batch_size=batch_size, distorted=True)
    images_test, labels_test = cifar.inputs(data_dir=data_dir, batch_size=batch_size, distorted=None)

    x = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
    y = tf.placeholder(tf.int32, [batch_size])

    model = cifar_10_model.Cifar10Model()
    train_op, res, loss = model.build(x, y, batch_size)
    top_k_op = tf.nn.in_top_k(res, y, 1)
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        tf.train.start_queue_runners()

        for step in range(max_steps):
            start_time = time.time()

            image_batch, label_batch = sess.run([images_train, labels_train])
            _, loss_value = sess.run([train_op, loss], feed_dict={x: image_batch, y: label_batch})
            duration = time.time() - start_time

            if step % 100 == 0:
                examples_per_sec = batch_size / duration
                sec_per_batch = float(duration)
                print("step %d,loss=%.2f(%.1f examples/sec;%.3f sec/batch)" % (
                    step, loss_value, examples_per_sec, sec_per_batch))

        num_batch = int(math.ceil(num_examples_for_eval / batch_size))  # math.ceil()函数用于求整
        true_count = 0
        total_sample_count = num_batch * batch_size

        # 在一个for循环里面统计所有预测正确的样例个数
        for j in range(num_batch):
            image_batch, label_batch = sess.run([images_test, labels_test])
            predictions = sess.run([top_k_op], feed_dict={x: image_batch, y: label_batch})
            true_count += np.sum(predictions)

        # 打印正确率信息
        print("accuracy = %.3f%%" % ((true_count / total_sample_count) * 100))
