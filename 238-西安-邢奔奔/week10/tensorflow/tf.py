import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
'''这里创建一个含有200个随机点，并转为列向量
noise为满足高斯分布的列向量'''
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

'''定义两个placeholder来存放数据，格式为float32，行数未知，列为1'''
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

'''定义中间层网络，输入为x数据，参数为10个，op操作为wx+b'''
weights_L1 = tf.Variable(tf.random.normal([1, 10]))
biase_L1 = tf.Variable(tf.zeros([1, 10]))
Wx_plus_b_L1 = tf.matmul(x, weights_L1) + biase_L1
L1 = tf.nn.tanh(Wx_plus_b_L1)  #这里是加入激活函数

'''定义输出层，输入为中间层输出，op依旧为wx+b，要满足矩阵运算法则 '''
weights_L2 = tf.Variable(tf.random.normal([10,1]))
biase_L2 = tf.Variable(tf.zeros([1, 1]))
Wx_plus_b_L2 = tf.matmul(L1, weights_L2) + biase_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)   #同样加入激活函数

'''定义损失函数'''
loss = tf.reduce_mean(tf.square(y - prediction))

'''定义反向传播算法（使用梯度下降法进行训练），学习率为0.1，损失函数为loss'''
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    #变量初始化
    sess.run(tf.global_variables_initializer())

    for i in range(2000):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})
        #进行2000次训练，然后进行推理
    prediction_values = sess.run(prediction, feed_dict={x: x_data})

    '''画图，来对比推理结果和原始数据的差别'''
    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, prediction_values, 'r-', lw=5)
    plt.show()
