import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data=np.linspace(-0.5,0.5,200)[:,np.newaxis]                          #使用numpy生成200个随机点
noise=np.random.normal(0,0.02,x_data.shape)
y_data=np.square(x_data)+noise
 

x=tf.placeholder(tf.float32,[None,1])                                   #定义两个placeholder存放输入数据
y=tf.placeholder(tf.float32,[None,1])
 

Weights_L1=tf.Variable(tf.random_normal([1,10]))                        #定义神经网络中间层
biases_L1=tf.Variable(tf.zeros([1,10]))                                 #加入偏置项
Wx_plus_b_L1=tf.matmul(x,Weights_L1)+biases_L1
L1=tf.nn.tanh(Wx_plus_b_L1)                                             #加入激活函数


Weights_L2=tf.Variable(tf.random_normal([10,1]))                        #定义神经网络输出层
biases_L2=tf.Variable(tf.zeros([1,1]))                                  #加入偏置项
Wx_plus_b_L2=tf.matmul(L1,Weights_L2)+biases_L2
prediction=tf.nn.tanh(Wx_plus_b_L2)                                     #加入激活函数


loss=tf.reduce_mean(tf.square(y-prediction))                            #定义损失函数（均方差函数）
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)        #定义反向传播算法（使用梯度下降算法训练）
 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())                         #变量初始化

    for i in range(2000):                                               #训练2000次
        sess.run(train_step,feed_dict={x:x_data,y:y_data})

    prediction_value=sess.run(prediction,feed_dict={x:x_data})          #获得预测值

    plt.figure()                                                        #画图
    plt.scatter(x_data,y_data)                                          #散点是真实值
    plt.plot(x_data,prediction_value,'r-',lw=5)                         #曲线是预测值
    plt.show()
