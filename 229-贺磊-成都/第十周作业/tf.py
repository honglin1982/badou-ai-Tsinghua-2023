import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 使用numpy生成200个随机点
# np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)
# 在间隔start和stop之间返回num个均匀间隔的数据(list类型)。
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]  # np.newaxis 放在哪个位置，就会给哪个位置增加维度
# x[:, np.newaxis] ，放在后面，会给列上增加维度
# x[np.newaxis, :] ，放在前面，会给行上增加维度

noise = np.random.normal(0, 0.02, x_data.shape)  # 用于生成服从正态分布的随机噪声，形状和x_data一样。
y_data = np.square(x_data) + noise

# 定义两个placeholder存放输入数据
x = tf.compat.v1.placeholder(tf.float32, [None, 1])  # 占位符，传入的第一个维度为任意维度，第二个维度为1
y = tf.compat.v1.placeholder(tf.float32, [None, 1])

# 定义神经网络中间层
Weights_L1 = tf.Variable(tf.random.normal([1, 10]))  # 创建了一个可训练的变量，并将其初始化为形状为[1,  10]的随机正态分布样本。
biases_L1 = tf.Variable(tf.zeros([1, 10]))  # 加入偏置项
Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1  # 内积之后，加上偏置
L1 = tf.nn.tanh(Wx_plus_b_L1)  # 加入激活函数

# 定义神经网络输出层
Weights_L2 = tf.Variable(tf.random.normal([10, 1]))
biases_L2 = tf.Variable(tf.zeros([1, 1]))  # 加入偏置项
Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)  # 加入激活函数

# 定义损失函数（均方差函数）
loss = tf.reduce_mean(tf.square(y - prediction))
# 定义反向传播算法（使用梯度下降算法训练）
train_step = tf.compat.v1.train.GradientDescentOptimizer(0.1).minimize(loss)  # 采用学习率为0.1的梯度下降法

with tf.compat.v1.Session() as sess:  # tf1创建会话操作，tf2不需要
    # 变量初始化q
    sess.run(tf.compat.v1.global_variables_initializer())
    # 训练2000次
    for i in range(2000):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})

    # 获得预测值
    prediction_value = sess.run(prediction, feed_dict={x: x_data})

    # 画图
    plt.figure()
    plt.scatter(x_data, y_data)  # 散点是真实值
    plt.plot(x_data, prediction_value, 'r-', lw=5)  # 曲线是预测值
    plt.show()
