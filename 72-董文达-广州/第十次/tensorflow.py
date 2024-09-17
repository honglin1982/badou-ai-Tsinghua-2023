import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None,1])

Wh = tf.Variable(tf.random_normal([1, 10]))
bh = tf.Variable(tf.zeros([1, 10]))
zh = tf.matmul(x, Wh) + bh
ah = tf.nn.tanh(zh)

Wo = tf.Variable(tf.random_normal([10, 1]))
bo = tf.Variable(tf.zeros([1, 1]))
zo = tf.matmul(ah, Wo) + bo
ao = tf.nn.tanh(zo)

loss = tf.reduce_mean(tf.square(y - ao))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(2000):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})

    ao_value = sess.run(ao, feed_dict={x: x_data})

    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, ao_value, 'r-', lw=5)
    plt.show()