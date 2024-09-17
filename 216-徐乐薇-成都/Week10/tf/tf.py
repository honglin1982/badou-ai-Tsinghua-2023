import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
 
#使用numpy生成200个随机点
x_data=np.linspace(-0.5,0.5,200)[:,np.newaxis]  #生成-0.5到0.5之间的200个点，[:,np.newaxis]将一维向量变成二维向量，变成200行1列的数据
noise=np.random.normal(0,0.02,x_data.shape)     #生成随机噪声,维度和x_data一样，均值为0，方差为0.02
y_data=np.square(x_data)+noise                  #y=x^2+噪声，即y_data是x_data的平方加上噪声
 
#定义两个placeholder存放输入数据
x=tf.placeholder(tf.float32,[None,1])           #None表示行数不确定，1表示列数为1
y=tf.placeholder(tf.float32,[None,1])           #None表示行数不确定，1表示列数为1
 
#定义神经网络中间层
Weights_L1=tf.Variable(tf.random_normal([1,10])) #定义权重，1行10列，正态分布随机数
biases_L1=tf.Variable(tf.zeros([1,10]))          #加入偏置项，1行10列，初始值为0
Wx_plus_b_L1=tf.matmul(x,Weights_L1)+biases_L1   #matmul是矩阵乘法，必须满足前列和后行相等
L1=tf.nn.tanh(Wx_plus_b_L1)                     #加入激活函数

#定义神经网络输出层
Weights_L2=tf.Variable(tf.random_normal([10,1])) #定义权重，10行1列
biases_L2=tf.Variable(tf.zeros([1,1]))          #加入偏置项，1行1列
Wx_plus_b_L2=tf.matmul(L1,Weights_L2)+biases_L2  #矩阵乘法，L1是上一层的输出，这一层的输入，Weights_L2是权重，biases_L2是偏置项
prediction=tf.nn.tanh(Wx_plus_b_L2)              #加入激活函数

#定义损失函数（均方差函数）
loss=tf.reduce_mean(tf.square(y-prediction))        #求均值，差的平方
#定义反向传播算法（使用梯度下降算法训练）
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss) #0.1表示学习率，即以0.1的效率来最小化loss，loss越小，预测值越接近真实值
 
with tf.Session() as sess:                          #创建会话
    #变量初始化
    sess.run(tf.global_variables_initializer())     #初始化变量
    #训练2000次
    for i in range(2000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data}) #训练2000次，每次都把x_data,y_data传入，因为x,y是placeholder，所以要用feed_dict传入
 
    #获得预测值
    prediction_value=sess.run(prediction,feed_dict={x:x_data})  #把x_data传入，得到预测值
 
    #画图
    plt.figure()
    plt.scatter(x_data,y_data)   #散点是真实值
    plt.plot(x_data,prediction_value,'r-',lw=5)   #曲线是预测值
    plt.show()
