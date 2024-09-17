# _*_ coding : utf-8 _*_
# @Time : 2023/7/21 8:40
# @Author : weixing
# @FileName : tensorflow_mnist
# @Project : cv

import numpy as np
import cv2
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.compat.v1.disable_eager_execution()

FLAGS = 1 # 0:训练，1：测试 2：验证

dataset_dir = "tensor_data&r/dataset"
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)
mnist = input_data.read_data_sets(dataset_dir, one_hot=True)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 28*28])
y_true = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])


W = tf.Variable(tf.zeros([28*28, 0]))
b = tf.Variable(tf.zeros([10]))

def weight_variable(shape):
  initial = tf.compat.v1.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#权重初始化
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

#第一层卷积
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#d第二层卷积
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


#全连接层
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.compat.v1.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


#输出层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

# y_pred = tf.nn.log_softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

y_pred = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#训练和评估模型
# cross_entropy = -tf.reduce_sum(y_true*tf.compat.v1.log(y_pred))
# train_step = tf.compat.v1.train.AdagradOptimizer(0.1).minimize(cross_entropy)
loss = tf.compat.v1.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))
train_step = tf.compat.v1.train.GradientDescentOptimizer(0.1).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

ckpt_dir = "tensor_data&r/model"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

# 标志变量不参与到训练中
global_step = tf.Variable(0, name='global_step', trainable=False)
saver = tf.compat.v1.train.Saver()

with tf.compat.v1.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print(ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)  # restore all variables
    else:
        tf.compat.v1.global_variables_initializer().run()

    start = global_step.eval() #get last global_step
    print("Start from:", start)

    # 训练
    for i in range(start, 2001): #接着从上次start的地方训练
      batch = mnist.train.next_batch(50)
      if i%100 == 0:
        train_accuracy = accuracy.eval(session=sess,feed_dict={
            x: batch[0], y_true: batch[1], keep_prob: 0.5})
        print("step %d, training accuracy %.4f"%(i, train_accuracy))
      train_step.run(session=sess, feed_dict={x: batch[0], y_true: batch[1], keep_prob: 0.5})
      global_step.assign(i).eval()  #i更新global_step.
      saver.save(sess, ckpt_dir + "/model.ckpt", global_step=global_step)

    # 测试
    test_acc = accuracy.eval(session=sess, feed_dict={
        x: mnist.test.images, y_true: mnist.test.labels, keep_prob: 0.5})
    print("test accuracy %.4f" % test_acc)
    if test_acc > 0.97:
        saver.save(sess, ckpt_dir + "/model.ckpt")

    img1 = cv2.imread("./testimg/test1.png", 0)
    x_test1 = np.array(img1.reshape((1,28*28)),dtype='float32')
    arr1 = [2]
    y_test1 = np.eye(10)[arr1]
    print("真实答案是：7，预测数字是：", np.argmax(sess.run(y_pred, feed_dict={x: x_test1, y_true: y_test1, keep_prob: 0.5})))

    img2 = cv2.imread("./testimg/test_2.png", 0)
    x_test2 = np.array(img2.reshape((1,28*28)), dtype='float32')
    arr2 = [2]
    y_test2 = np.eye(10)[arr2]
    print("真实答案是：2，预测数字是：",np.argmax(sess.run(y_pred, feed_dict={x: x_test2, y_true: y_test2, keep_prob: 0.5})))

    # for i in range(10):
    #     # 每次测试一张图片 [0,0,0,0,0,1,0,0,0,0]
    #     x_test, y_test = mnist.test.next_batch(1)
    #     print("第%d张图片，手写数字图片目标是:%d, 预测结果是:%d" % (
    #         i,
    #         np.argmax(y_test),
    #         np.argmax(sess.run(y_pred, feed_dict={x: x_test, y_true: y_test, keep_prob: 0.5}))
    #     ))
