import tensorflow as tf

input1 = tf.placeholder(tf.float32) #定义占位符
input2 = tf.placeholder(tf.float32) #定义占位符
output = tf.multiply(input1, input2) #定义乘法运算，multiply是元素乘法

with tf.Session() as sess:
  print(sess.run([output], feed_dict={input1:[7], input2:[2.]})) #feed_dict 是一个字典，给出每个用到的占位符的取值, 以及对应的取值.input1:[7]表示给input1赋值为7,input2:[2.]表示给input2赋值为2.0

# 输出: [array([ 14.], dtype=float32)] # 7*2=14
