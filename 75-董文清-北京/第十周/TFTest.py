import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#随机生成 输入数据
xdata = np.linspace(-0.5, 0.5, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.02, xdata.shape)
ydata = np.square(xdata) + noise
#提前声明 占位数据
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])
#定义神经网络中间层
WeightsL1 = tf.Variable(tf.random_normal([1, 10]))
BiasesL1 = tf.Variable(tf.random_normal([1, 10]))
WplusbL1 = tf.matmul(x, WeightsL1) + BiasesL1
L1 = tf.nn.tanh(WplusbL1)
#定义神经网络输出层
WeightsL2 = tf.Variable(tf.random_normal([10, 1]))
BiasesL2 = tf.Variable(tf.random_normal([1, 1]))
WplusbL2 = tf.matmul(L1, WeightsL2) + BiasesL2
predication = tf.nn.tanh(WplusbL2)
#定义损失函数与反向传播
loss = tf.reduce_mean(tf.square(y - predication))
trainStep = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#Session run
with tf.Session() as sess:
    #初始化
    sess.run(tf.compat.v1.global_variables_initializer())
    #训练
    for i in range(2000) : sess.run(trainStep, feed_dict={x : xdata, y : ydata})
    #预测
    predicationValue = sess.run(predication, feed_dict={x : xdata})

    plt.figure()
    plt.scatter(xdata, ydata)
    plt.plot(xdata, predicationValue, 'r-', lw=5)
    plt.show()

