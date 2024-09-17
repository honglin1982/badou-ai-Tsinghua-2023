# 在tensorflow1.x框架基础上实现简单神经网络
# 如何改成适应2.x版本的代码？

import tensorflow.compat.v1 as tf

import numpy as np
import matplotlib.pylab as plt

tf.disable_eager_execution()
# 使用numpy生成200个随机点
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

# 定义两个placeholder存放输入数据
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 定义神经网络中间层 w1x+b1=L1
Weights_L1 = tf.Variable(tf.random_normal([1,10]))
biases_L1 = tf.Variable(tf.zeros([1,10]))
Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1
L1 = tf.nn.tanh(Wx_plus_b_L1)

# 定义神经网络输出层 w2L1+b2=预测值
Weights_L2 = tf.Variable(tf.random_normal([10, 1]))
biases_L2 = tf.Variable(tf.zeros([1,1]))
Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)

# 定义损失函数（均方差，适用于回归）
loss = tf.reduce_mean(tf.square(y - prediction))
# 定义反向传播算法（梯度下降）
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.session() as sess:
    # 参数初始化
    sess.run(tf.global_variables_initializer())
    # 训练2000次
    for i in range(2000):
        sess.run(train_step, feed_dict={x:x_data, y:y_data})

    # 获得预测值
    prediction_value = sess.run(prediction, feed_dict={x:x_data})

    #画图
    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, prediction_value, 'r-', lw=5)  # r-红色实线，lw线宽
    plt.show()

