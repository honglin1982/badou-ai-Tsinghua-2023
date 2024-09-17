import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

'''
tf实现简单神经网络
pytorch实现手写数字识别
'''


train_data = np.linspace(-1, 1, 100)[:, np.newaxis]
train_label = np.square(train_data) + np.random.normal(0, 0.01, train_data.shape)

input_size = 1
hidden_size = 10
output_size = 1

x = tf.placeholder(dtype=tf.float32, shape=(None, input_size))
y = tf.placeholder(dtype=tf.float32, shape=(None, output_size))
w1 = tf.Variable(tf.random.normal([input_size, hidden_size]), name="w1")
b1 = tf.Variable(tf.random.normal([1, hidden_size]), name="b1")
w2 = tf.Variable(tf.random.normal([hidden_size, output_size]), name="w2")
b2 = tf.Variable(tf.random.normal([1, output_size]), name="b2")

output1 = tf.nn.tanh((tf.matmul(x, w1) + b1))
output2 = tf.nn.tanh(tf.matmul(output1, w2) + b2)

loss = tf.reduce_mean(tf.square(y - output2))

train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(2000):
    sess.run(train, feed_dict={x: train_data, y: train_label})
predict = sess.run(output2, feed_dict={x: train_data})
sess.close()

plt.scatter(train_data, train_label)
plt.plot(train_data, predict, c="r")
plt.show()