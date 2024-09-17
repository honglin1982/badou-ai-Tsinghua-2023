import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#生成数据,[:, np.newaxis]用于将数组从二维降为一维，也就是将列向量转换为行向量
x_data = np.linspace(-0.5,0.5,200)[:,np.newaxis]
other_data = np.random.normal(0,0.02,x_data.shape)
y_data = np.square(x_data) + other_data

#定义输入
x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])

#生成L1网络层
Weights_L1 = tf.Variable(tf.random_normal([1,10]))
Biases_L1 = tf.Variable(tf.zeros([1,10]))  #加入偏置项
Wx_plus_b_L1 = tf.matmul(x,Weights_L1) + Biases_L1 #y=wx+b
L1 = tf.nn.tanh(Wx_plus_b_L1)                  #激活函数

#生成L2网络层
Weights_L2 = tf.Variable(tf.random_normal([10,1])) #列数要跟前一层输出行数一样
Biases_L2 = tf.Variable(tf.zeros([1,1]))  #加入偏置项
Wx_plus_b_L2 = tf.matmul(L1,Weights_L2) + Biases_L2 #y=wx+b
prediction = tf.nn.tanh(Wx_plus_b_L2)                  #激活函数

#定义损失函数
loss = tf.reduce_mean(tf.square(y-prediction))
#定义反向传播算法（使用梯度下降算法训练）
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    #初始化
    sess.run(tf.global_variables_initializer())
    #训练2000次
    for i in range(2000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})
    #获得预测值
    prediction_values = sess.run(prediction,feed_dict = {x:x_data})

#画图
plt.figure()
plt.scatter(x_data,y_data)          #散点是真实值
plt.plot(x_data,prediction_values,'r-',lw=5)   #曲线是预测值
plt.show()
