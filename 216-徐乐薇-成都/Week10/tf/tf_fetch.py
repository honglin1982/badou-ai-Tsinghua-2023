import tensorflow as tf

input1 = tf.constant(3.0)  #定义常量, 一旦定义后, 值就不能改变了
input2 = tf.constant(2.0)  #定义常量, 一旦定义后, 值就不能改变了
input3 = tf.constant(5.0)  #定义常量, 一旦定义后, 值就不能改变了
intermed = tf.add(input2, input3) #定义变量, 在 session 中需要初始化
mul = tf.multiply(input1, intermed) #定义变量, 在 session 中需要初始化

with tf.Session() as sess: #创建会话
  result = sess.run([mul, intermed])   #需要获取的多个 tensor 值，在 op 的一次运行中一起获得（而不是逐个去获取 tensor）。
  print(result)
