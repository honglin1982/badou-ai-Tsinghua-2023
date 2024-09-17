import tensorflow as tf


input1 = tf.constant(3)
input2 = tf.constant(2)
input3 = tf.constant(5)
intermed = tf.add(input2,input3)
mul = tf.multiply(input1,input2)

'''可以在op的一次运行中，获取多个tensor，只要定义了就行'''
with tf.Session() as sess:
    resu = sess.run([mul,intermed])
    print(resu)