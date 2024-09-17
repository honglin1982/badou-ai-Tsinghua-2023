import tensorflow as tf
'''tf的pyhton有一个默认图，op构造器可以为其增加节点
这里创建了matrix1是一个1X2矩阵，matrix2是一个2X1矩阵，op为矩阵乘法，op增加到默认图中'''
matrix1 = tf.constant([[3,3]])

matrix2 = tf.constant([[2],[2]])

product = tf.matmul(matrix1,matrix2)

'''这里启动图，但是一定要关闭，如果使用下面的with方法，就无须考虑关闭动作'''
# sess = tf.Session()
#
# result = sess.run(product)
# print(result)
#
# sess.close()

with tf.Session() as sess:
    result = sess.run(product)
    print(result)
