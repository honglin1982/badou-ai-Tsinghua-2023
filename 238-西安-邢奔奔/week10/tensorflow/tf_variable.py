import tensorflow as tf

#创建一个变量，初始化为0
state = tf.Variable(0,name='counter')
#创建一个op作用是使得state增加1
one = tf.constant(1)
new_value = tf.add(state,one)
#tf.assign用来将state的值更新为new_value的值
update =tf.assign(state,new_value)

#初始化定义的变量
init_op = tf.global_variables_initializer()

#启动图
with tf.Session() as sess:
    sess.run(init_op)
    for i in range(5):
        sess.run(update)
        print(sess.run(state))
'''可以这么理解：
这是一个倒过程，运行update会找到new_value，就会进行增加操作'''