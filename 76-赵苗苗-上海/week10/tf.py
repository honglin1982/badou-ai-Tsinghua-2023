#描述了一个具有一个隐藏层的前馈神经网络，用于拟合输入数据和输出数据之间的非线性关系
#通过训练模型，可以获得预测值，并将预测结果与真实值进行可视化比较
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#1、使用 numpy 生成随机点作为输入数据和加入噪声的输出数据
x_data=np.linspace(-0.5,0.5,200)[:,np.newaxis]
noise=np.random.normal(0,0.02,x_data.shape)
y_data=np.square(x_data)+noise

#2、定义了两个占位符 ​x​ 和 ​y​ 来传递输入数据和标签数据
x=tf.compat.v1.placeholder(tf.float32,[None,1])
y=tf.compat.v1.placeholder(tf.float32,[None,1])

#3、定义了神经网络的中间层，包括权重、偏置和激活函数（tanh）
weights_L1=tf.Variable(tf.random.normal([1,10]))  #w1
biases_L1=tf.Variable(tf.zeros([1,10]))           #b1
Wx_plus_b_L1=tf.matmul(x,weights_L1)+biases_L1    #矩阵乘
L1=tf.tanh(Wx_plus_b_L1)                          #加入激活函数

#4、定义了神经网络的输出层，同样包括权重、偏置和激活函数（tanh）
weights_L2=tf.Variable(tf.random.normal([10,1]))  #w2
biases_L2=tf.Variable(tf.zeros([1,1]))            #b2
Wx_plus_b_L2=tf.matmul(L1,weights_L2)+biases_L2    #矩阵乘
prediction=tf.nn.tanh(Wx_plus_b_L2)                       #加入激活函数

#5、定义了损失函数（均方差函数）来衡量预测值与真实值之间的差距
loss=tf.reduce_mean(tf.square(y-prediction))

#6、实现基于梯度下降算法的神经网络训练过程，初始化变量、执行训练步骤、获取预测结果
#执行参数loss更新，tf.compat.v1.train.GradientDescentOptimizer(0.1)​ 创建了一个梯度下降优化器对象，指定学习率为0.1
train_step=tf.compat.v1.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.compat.v1.Session() as sess:
    #初始化所有全局变量，包括权重和偏置项
    sess.run(tf.compat.v1.global_variables_initializer())
    #训练2000次
    for i in range(20000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})
    #获取模型基于输入数据x_data的预测值
    prediction_value=sess.run(prediction,feed_dict={x:x_data})
    #画图
    plt.figure()
    plt.scatter(x_data,y_data)  #真实——散点
    plt.plot(x_data,prediction_value,'r-',lw=5) #预测——曲线
    plt.show()

