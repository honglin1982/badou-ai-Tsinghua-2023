import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 生成上下区间的随机点 np.newaxis转为二维数组
x_data = np.linspace(-0.5, 0.5, 1000)[:, np.newaxis]
random_data = np.random.normal(0, 0.02, x_data.shape)
y_data = x_data + random_data

# 定义两个接收变量
inputs = tf.placeholder(tf.float32, [None, 1])
result = tf.placeholder(tf.float32, [None, 1])

# 构建隐藏层 10个节点
weights_l1 = tf.Variable(tf.random_normal([1, 50]))
biases_l1 = tf.Variable(tf.zeros([1, 50]))
# ax+b
w_plus_b_l1 = tf.matmul(inputs, weights_l1) + biases_l1
# 过激活函数
l1 = tf.nn.tanh(w_plus_b_l1)

# 构建输出层
weights_l2 = tf.Variable(tf.random_normal([50, 1]))
biases_l2 = tf.Variable(tf.zeros([1, 1]))
# ax+b
w_plus_b_l2 = tf.matmul(l1, weights_l2) + biases_l2
# 过激活函数
prediction = tf.nn.tanh(w_plus_b_l2)

# 计算损失函数
loss = tf.reduce_mean(tf.square(result - prediction))

# 反向传播 计算权重和biase
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as session:
    # 变量初始化
    session.run(tf.global_variables_initializer())
    # 训练2000次
    for i in range(2000):
        session.run(train_step, feed_dict={inputs: x_data, result: y_data})

    # 获得预测值
    prediction_value = session.run(prediction, feed_dict={inputs: x_data})

    # 画图
    plt.figure()
    # 散点是真实的值
    plt.scatter(x_data, y_data)
    # 曲线是预测值
    plt.plot(x_data, prediction_value, 'r-', lw=5)
    plt.show()
