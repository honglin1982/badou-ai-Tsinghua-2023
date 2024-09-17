import tensorflow as tf


a = tf.constant([10,20,40],name='a',dtype=tf.float32)
b = tf.Variable(tf.random_uniform([3]),name='b')
output = tf.add_n([a,b],name='add')

writer = tf.summary.FileWriter('logs',tf.get_default_graph())
writer.close()