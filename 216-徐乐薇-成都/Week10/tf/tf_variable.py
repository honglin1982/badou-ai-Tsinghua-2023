import tensorflow as tf

# 创建一个变量, 初始化为标量 0.
state = tf.Variable(0, name="counter")     #变量，可变，可训练

# 创建一个 op, 其作用是使 state 增加 1
one = tf.constant(1)                       #常量，不可变，不可训练
new_value = tf.add(state, one)             #add是加法
update = tf.assign(state, new_value)       #assign是赋值

# 启动图后, 变量必须先经过`初始化` (init) op 初始化,
# 首先必须增加一个`初始化` op 到图中.
init_op = tf.global_variables_initializer() #初始化所有变量

# 启动图, 运行 op
with tf.Session() as sess:
  # 运行 'init' op
  sess.run(init_op)                       #初始化所有变量
  # 打印 'state' 的初始值
  print("state",sess.run(state))          #打印state的初始值
  # 运行 op, 更新 'state', 并打印 'state'
  for _ in range(5):                      #循环5次
    sess.run(update)                      #运行op,更新state
    print("update:",sess.run(state))      #打印state
