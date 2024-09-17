import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution()  # 兼容tf2.0以上的版本

# 使用numpy生成200个随机点
x_data = np.array(np.linspace(-0.5, 0.5, 200))[:, np.newaxis]  # input: (200, 1)
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise  # target: (200, 1)

# 定义两个placeholder存放输入数据
x = tf.compat.v1.placeholder(tf.float32, shape=(None, 1), name="x")
y = tf.compat.v1.placeholder(tf.float32, shape=(None, 1), name="y")

# 定义神经网络中间层
WeightL1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1, 10]))
biasesL1 = tf.compat.v1.Variable(tf.zeros([1, 10]))  # 加入偏置项
WX_plus_b_l1 = tf.matmul(x, WeightL1) + biasesL1  # shape(200, 10)
L1 = tf.nn.tanh(WX_plus_b_l1)  # 加入激活函数

# 定义神经网络输出层
WeightL2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([10, 1]))
biasesL2 = tf.compat.v1.Variable(tf.zeros([1, 1]))  # 加入偏置项
WX_plus_b_l2 = tf.matmul(L1, WeightL2) + biasesL2  # shape(200, 1)
prediction = tf.nn.tanh(WX_plus_b_l2)  # 加入激活函数

# 定义损失函数（均方差函数）
loss = tf.reduce_mean(tf.square(y - prediction))

# 定义反向传播算法（使用梯度下降算法训练）
train_step = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

with tf.compat.v1.Session() as sess:
    # 变量初始化
    sess.run(tf.compat.v1.global_variables_initializer())
    # 训练2000次
    for i in range(2000):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})
    # 获得预测值
    prediction_value = sess.run(prediction, feed_dict={x: x_data})
    # 画图
    plt.figure()
    plt.scatter(x_data, y_data, c='orange')
    plt.plot(x_data, prediction_value, 'r-', lw=5)  # 曲线是预测值
    plt.show()



