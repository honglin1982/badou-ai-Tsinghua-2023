import tensorflow as tf

## 定义一个计算图，包含加法和乘法
i1 = tf.constant(2.0)
i2 = tf.constant(4.0)
i3 = tf.constant(8.0)
intermed = tf.add(i2, i3)
o = tf.multiply(i1, intermed)

with tf.compat.v1.Session() as sess:
    #! 需要获取的多个tensor值，在op的一次运行中一起获得（而不是逐个去获取）
    result = sess.run([intermed, o])
    print("intermed: ", result[0])
    print("output: ", result[1])