import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

## 创建200个数
#? 创建[-0.5, 0.5]期间的200个等间距的数，再通过[:, np.newaxis]将其从
#? (200,)的一维数组转换成(200, 1)的二维数组
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
noise = np.random.normal(0, 0.02, (200, 1))
#? y = x**2 + noise
y_data = np.square(x_data) + noise

## 定义两个placeholder存放输入数据，形状为[None, 1]
x = tf.compat.v1.placeholder(tf.float32, [None, 1])
y = tf.compat.v1.placeholder(tf.float32, [None, 1])

## 定义神经网络中间层
#? 定义第一层权重矩阵，形状为[1, 10]，均值为0，标准差为1
weights_l1 = tf.Variable(tf.random.normal([1, 10]))
#? 定义第一层偏置项，形状也为[1, 10]，本例设为0
biases_l1 = tf.Variable(tf.zeros([1, 10]))
#? 定义计算过程
Wx_plus_b_l1 = tf.matmul(x, weights_l1) + biases_l1
#? 加入激活函数
L1 = tf.nn.tanh(Wx_plus_b_l1)

## 定义神经网络输出层，同理，但是形状需要根据输出需要进行调整
weights_l2 = tf.Variable(tf.random.normal([10, 1]))
biases_l2 = tf.Variable(tf.zeros([1, 1]))
Wx_plus_b_l2 = tf.matmul(L1, weights_l2) + biases_l2
prediction = tf.nn.tanh(Wx_plus_b_l2)

## 定义损失函数，均方差函数
loss = tf.reduce_mean(tf.square(y-prediction))
## 定义反向传播算法，梯度下降法，指定学习率为0.1，损失函数为loss
train_step = tf.compat.v1.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.compat.v1.Session() as sess:
    #! 变量init
    sess.run(tf.compat.v1.global_variables_initializer())
    ## 训练2000次
    for _ in range(2000):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})
    ## 推理获得预测值
    prediction_values = sess.run(prediction, feed_dict={x: x_data})
    
    #> 画图
    plt.figure()
    ## 原始数据散点图
    plt.scatter(x_data, y_data)
    ## 预测曲线
    plt.plot(x_data, prediction_values, 'r-', lw=3)
    plt.show()