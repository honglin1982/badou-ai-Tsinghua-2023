import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# matrix1 = tf.constant([[4.,3]])
# matrix2 = tf.constant([[2.],[6]])
# product = tf.matmul(matrix1,matrix2)
#
# sess = tf.Session()
# result=sess.run(product)
# print(result)
# sess.close()


# state=tf.Variable(0,name="counter")
# one = tf.constant(1)
# update = tf.assign_add(state,one)
# init_op = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init_op)
#     print("state",sess.run(state))
#     for _ in range(5):
#         sess.run(update)
#         print("update",sess.run(state))


# input1 = tf.constant(3.)
# input2 = tf.constant(2.)
# input3 = tf.constant(5.)
# intermed = tf.add(input2,input3)
# mul = tf.multiply(intermed,input1)
# with tf.Session() as sess:
#     result=sess.run([mul,intermed,mul])
#     print(result)


# input1 = tf.placeholder(tf.float32)
# input2 = tf.placeholder(tf.float32)
# output = tf.multiply(input1,input2)
# with tf.Session() as sess:
#     print(sess.run([output],{input1:[7],input2:[2.]}))


# # 新建一个graph.
# with tf.device('/cpu:0'):
#   a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
#   b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
# c = tf.matmul(a, b, name = "mul")
# print("tesnor: ", c)
#
# # 新建session with log_device_placement并设置为True.
# sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
# print (sess.run(c))
# sess.close()
with tf.device('/cpu:0'):
    x_data = np.linspace(-0.5,0.5,200)[:,np.newaxis]
    noise = np.random.normal(0,0.02,x_data.shape)
    y_data = np.square(x_data)+noise

    x=tf.placeholder(tf.float32,[None,1])
    y=tf.placeholder(tf.float32,[None,1])

    Weights_L1=tf.Variable(tf.random_normal([1,10]))
    biases_L1=tf.Variable(tf.zeros([1,10]))
    Wx_plus_b_L1=tf.matmul(x,Weights_L1)+biases_L1
    L1=tf.nn.tanh(Wx_plus_b_L1)

    Weights_L2=tf.Variable(tf.random.normal([10,1]))
    biases_L2=tf.Variable(tf.zeros([1,1]))
    Wx_plus_b_L2=tf.matmul(L1,Weights_L2)+biases_L2
    prediction=tf.nn.tanh(Wx_plus_b_L2)
    loss=tf.reduce_mean(tf.square(y-prediction))
    train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})
    prediction_value=sess.run(prediction,feed_dict={x:x_data})

    plt.figure()
    plt.scatter(x_data,y_data)
    plt.plot(x_data,prediction_value,'-r',lw=5)
    plt.show()
writer=tf.summary.FileWriter('logs', tf.get_default_graph())
writer.close()

# # 定义一个计算图，实现两个向量的加法操作
# # 定义两个输入，a为常量，b为变量
# a=tf.constant([10.0, 20.0, 40.0], name='a')
# b=tf.Variable(tf.random_uniform([3]), name='b')
# output=tf.add_n([a,b], name='add')
#
# # 生成一个具有写权限的日志文件操作对象，将当前命名空间的计算图写进日志中
# writer=tf.summary.FileWriter('logs', tf.get_default_graph())
# writer.close()
