import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# 矩阵相乘
matrix1 = tf.constant([[2, 3]])
matrix2 = tf.constant([[2], [3]])

op1 = tf.matmul(matrix1, matrix2)

with tf.compat.v1.Session() as sess:
    result = sess.run(op1)
    print(result)

# 变量的计数操作
state = tf.Variable(0, name='counter')
v1 = tf.constant(1)
jiafa = tf.add(state, v1)
update = tf.compat.v1.assign(state, jiafa)
init_op = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess1:
    sess1.run(init_op)
    for _ in range(5):
        r = sess1.run(update)
        print(r)
        a = state.eval()
print(state)

# tf graph
# 新建一个graph.
with tf.device("/cpu:0"):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name="a")
    print(a)
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name="b")
    print(b)

c = tf.matmul(a, b, name="mul")
print("tensor: ", c)
'''
如果你指定的设备不存在, 你会收到 InvalidArgumentError 错误提示。
可以在创建的 session 里把参数 allow_soft_placement 设置为 True, 这样 tensorFlow 会自动选择一个存在并且支持的设备来运行 operation.
'''
# sess2 = tf.compat.v1.Session()
sess2 = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True))
print(sess2.run(c))
sess2.close()

# fetch
input1 = tf.constant(2)
input2 = tf.constant(5)
input3 = tf.constant(10)
input_add_func = tf.add(input2, input3)
mul = tf.multiply(input1, input_add_func)

with tf.compat.v1.Session() as sess:
    result = sess.run([mul, input_add_func])
    print(f'--result: {result}')

# feed
input1 = tf.compat.v1.placeholder(tf.float32)
input2 = tf.compat.v1.placeholder(tf.float32)
output = tf.multiply(input1, input2)
with tf.compat.v1.Session() as sess:
    r = sess.run([output], feed_dict={input1: [7], input2: [2.]})
    print(r)


