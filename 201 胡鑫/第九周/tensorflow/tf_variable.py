import tensorflow as tf

# ! 变量op在开启计算图后，应该首先加入一个（init）op
# # 创建一个变量op，初始化为标量0
state = tf.Variable(0)
# # 创建一个op，其作用是使state增加1
one = tf.constant(1)
new_state = tf.add(state, one)
# ? 将new_state的值赋值给state
update = tf.assign(state, new_state)

# # 创建一个初始化变量的op
init_op = tf.global_variables_initializer()

# # 开始
with tf.compat.v1.Session() as sess:
    # 初始化变量op
    sess.run(init_op)
    # 打印state初始值
    print("state: ", sess.run(state))
    # 更新
    for _ in range(5):
        print("update: ", sess.run(update))
    