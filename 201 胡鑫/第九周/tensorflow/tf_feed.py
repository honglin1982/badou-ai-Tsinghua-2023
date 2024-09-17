import tensorflow as tf

## 创建一个计算图，在初始化输入值的时候使用placehoder
i1 = tf.placeholder(tf.float32)
i2 = tf.placeholder(tf.float32)
output = tf.multiply(i1, i2)

with tf.compat.v1.Session() as sess:
    ## 在sess.run中使用feed_dict参数传递一个字典，这个字典中初始化输入值
    print(sess.run([output], feed_dict={i1: [7], i2: [3.0]}))