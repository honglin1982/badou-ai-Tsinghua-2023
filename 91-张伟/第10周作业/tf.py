import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1，生成随机数
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis] #生成200个二维数据
noise = np.random.normal(0, 0.02, x_data.shape)#随机抽取样本形成噪音
y_data = np.square(x_data) + noise

# 2.定义两个placeholder存放输入数据
x = tf.placeholder(tf.float32, [None, 1])#[None, 1]多行1列
y = tf.placeholder(tf.float32, [None, 1])

# 3.定义中间层
weight_l1 = tf.Variable(tf.random_normal([1,10]))
biases_l1 = tf.Variable(tf.zeros([1, 10])) #偏置量
wx_plus_b_l1 = tf.matmul(x, weight_l1) + biases_l1#得到信号总和
l1 = tf.nn.tanh(wx_plus_b_l1)#设置激活函数

# 4.定义输出层
weight_l2 = tf.Variable(tf.random_normal([10,1]))
biases_l2 = tf.Variable(tf.zeros([1, 1]))
wx_plus_b_l2 = tf.matmul(l1, weight_l2) +biases_l2
prediction = tf.nn.tanh(wx_plus_b_l2)

# 5.定义损失函数（均值方差）
loss = tf.reduce_mean(tf.square(y - prediction))

# 6.定义反向传播算法，使用梯度下降算法训练
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
with tf.Session() as sess:
    # 变量初始化
    sess.run(tf.global_variables_initializer())
    # 训练
    for i in range(2500):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})

    # 获得预测值
    prediction_value = sess.run(prediction, feed_dict={x: x_data})

    # 画图
    plt.figure()
    plt.scatter(x_data, y_data)  # 散点是真实值
    plt.plot(x_data, prediction_value, 'r-', lw=5)  # 曲线是预测值
    plt.show()


