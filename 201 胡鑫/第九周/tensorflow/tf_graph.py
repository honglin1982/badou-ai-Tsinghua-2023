import tensorflow as tf

# 创建一个常量op（1x2的矩阵）
input1 = tf.constant([[2., 2.]])
# 创建第二个常量op（2x1的矩阵）
input2 = tf.constant([[2.], [2.]])
# 创建一个矩阵乘法op，该op的返回值为矩阵相乘的结果
mul = tf.matmul(input1, input2)
# 开启默认计算图
sess = tf.compat.v1.Session()
# ! 直接将矩阵乘法的返回值作为参数传入run方法，表示获取这个方法的结果
res = sess.run(mul)
print(res)
# 关闭计算图
sess.close()