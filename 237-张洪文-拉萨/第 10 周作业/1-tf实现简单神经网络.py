import os
import tempfile
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# feed解析
def tf_feed(input_1, input_2):
    """
    在TensorFlow中，feed用于给占位符（Placeholder）提供数据。
    占位符是在计算图构建阶段没有具体值的节点，相当于在运行时才决定的变量。
    你可以通过feed_dict参数在运行会话时，动态地传递数据给占位符。
    """
    input1 = tf.placeholder(tf.float32)
    input2 = tf.placeholder(tf.float32)
    output = tf.multiply(input1, input2)  # 创建一个执行元素级乘法的操作节点

    with tf.Session() as sess:
        result = sess.run(output, feed_dict={input1: input_1, input2: input_2})
        print(result)

# fetch解析
def tf_fetch():
    input1 = tf.constant(3.0)
    input2 = tf.constant(4.0)
    input3 = tf.constant(5.0)
    intermed = tf.add(input1, input2)
    mul = tf.multiply(input3, intermed)

    with tf.Session() as sess:
        # 需要获取的多个 tensor 值，在 op 的一次运行中一起获得（而不是逐个去获取 tensor）
        result = sess.run([mul, intermed])
        print(result)

# graph解析
def tf_graph():
    """
    TensorFlow Python 库有一个默认图 (default graph), op 构造器可以为其增加节点.
    这个默认图对许多程序来说已经足够用了.
    创建一个常量 op, 产生一个 1x2 矩阵. 这个 op 被作为一个节点加到默认图中.
    """
    # 构造器的返回值代表该常量 op 的返回值. 1x2 矩阵
    matrix1 = tf.constant([[3., 3.]])
    # 创建另外一个常量 op, 产生一个 2x1 矩阵.
    matrix2 = tf.constant([[2.], [2.]])
    # 创建一个矩阵乘法 matmul op , 把 'matrix1' 和 'matrix2' 作为输入.
    # 返回值 'product' 代表矩阵乘法的结果.
    product = tf.matmul(matrix1, matrix2)
    '''
    默认图现在有三个节点, 两个 constant() op, 和一个matmul() op. 
    为了真正进行矩阵相乘运算, 并得到矩阵乘法的结果, 必须在会话里启动这个图.
    启动图的第一步是创建一个 Session 对象, 如果无任何创建参数, 会话构造器将启动默认图.
    '''
    # 启动默认图.
    sess = tf.Session()

    """ 
    调用 sess 的 'run()' 方法来执行矩阵乘法 op, 传入 'product' 作为该方法的参数. 
    'product' 代表了矩阵乘法 op 的输出, 传入它是向方法表明, 我们希望取回矩阵乘法 op 的输出.
    整个执行过程是自动化的, 会话负责传递 op 所需的全部输入. op 通常是并发执行的.
    函数调用 'run(product)' 触发了图中三个 op (两个常量 op 和一个矩阵乘法 op) 的执行.
    """
    # 返回值 'result' 是一个 numpy `ndarray` 对象.
    result = sess.run(product)
    print(result)  # ==> [[ 12.]]


    '''
    session对象在使用完后需要关闭以释放资源. 除了显式调用 close 外, 也可以使用 “with” 代码块来自动完成关闭动作.
    with tf.Session() as sess:
      result = sess.run([product])
      print (result)
    '''
    sess.close()

# cpu设备指定解析
def tf_graph_cpu():
    # 创建会话时设置CPU设备配置
    config = tf.ConfigProto(device_count={'CPU': 8})
    sess = tf.Session(config=config)
    # 查看当前系统支持的所有物理设备
    physical_devices = sess.list_devices()
    print("支持的物理设备:")
    for device in physical_devices:
        print(device)
    # 关闭会话
    sess.close()
    """
    在实现上, TensorFlow 将图形定义转换成分布式执行的操作, 以充分利用可用的计算资源(如 CPU或 GPU).
    一般你不需要显式指定使用 CPU 还是 GPU, TensorFlow 能自动检测.
    如果检测到 GPU, TensorFlow 会尽可能地利用找到的第一个 GPU 来执行操作.
    如果你的系统里有多个 GPU, 那么 ID 最小的 GPU 会默认使用。
    如果你想要手动指派设备, 你可以用 with tf.device 创建一个设备环境。
    这个环境下的 operation 都统一运行在环境指定的设备上.
    """
    # 使用with tf.device('/cpu:0')指定操作运行在CPU上
    with tf.device('/cpu:2'):
        # 创建两个常量张量a和b
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    # 使用tf.matmul()进行矩阵相乘操作，结果保存在张量c中
    c = tf.matmul(a, b, name="mul")
    print("Tensor c: ", c)  # 输出张量c的信息

    """
    如果你指定的设备不存在, 你会收到 InvalidArgumentError 错误提示。
    可以在创建的 session 里把参数 allow_soft_placement 设置为 True, 
    这样 tensorFlow 会自动选择一个存在并且支持的设备来运行 operation。
    log_device_placement=True表示在运行时打印每个操作所在的设备信息。
    """
    # 创建会话并运行计算图
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        # 执行矩阵相乘操作，输出结果
        result = sess.run(c)
        print(f"Result: {result}")

# tensorboard 服务解析
def tf_tensorboard():
    # 定义一个计算图，实现两个向量的减法操作：定义两个输入，a为常量，b为变量
    a = tf.constant([10.0, 20.0, 40.0], name='a')
    b = tf.Variable(tf.random_uniform([3]), name='b')
    output = tf.add_n([a, b], name='add')

    # 生成一个具有写权限的日志文件操作对象，将当前命名空间的计算图写进日志中:该路径为启动tensorboard的路径
    writer = tf.summary.FileWriter(r'E:\ProgramFiles\CodeEnvironment\venv\logs', tf.get_default_graph())
    writer.close()

    """
    启动tensorboard服务（在命令行启动）:
    1.无环境变量：python.exe -m tensorboard.main --logdir C:\\Users\wen\logs
    2.有环境变量：tensorboard --logdir logs
    """
    # 启动tensorboard服务后，复制地址并在本地浏览器中打开，

# variable 解析
def tf_variable():
    # 创建一个变量: 初始化为标量 0.
    state = tf.Variable(0, name="counter")
    # 创建一个 op, 其作用是使 state 增加 1
    one = tf.constant(1)  # 创建了一个常量one，其值为1
    new_value = tf.add(state, one)  # 将state和one相加
    update = tf.assign(state, new_value)  # tf.assign()将state的值更新为new_value
    # 启动图后, 变量必须先经过`初始化` (init) op 初始化,
    # 首先必须增加一个`初始化` op 到图中.
    init_op = tf.global_variables_initializer()

    # 启动图, 运行 op
    with tf.Session() as sess:
        # 运行 'init' op
        sess.run(init_op)
        # 打印 'state' 的初始值
        print("state:", sess.run(state))
        # 运行 op, 更新 'state', 并打印 'state'
        for _ in range(5):
            sess.run(update)
            print("update:", sess.run(state))


# 1、使用 TensorFlow 构建了一个简单的神经网络
def tensorflow_1():
    # 1. 生成训练数据: 生成一个等间隔的数列, 并转为列向量（200，1）
    x_data = np.linspace(start=-0.5, stop=0.5, num=200)[:, np.newaxis]  # np.newaxis用于增加维度
    noise_var = np.random.normal(0, 0.02, x_data.shape)  # 生成正态分布的噪声值
    y_data = np.square(x_data) + noise_var  # y = x^2 + b

    # 2. 定义TensorFlow占位符（placeholder）:shape的值根据训练数据格式决定，1 表特征数为1
    x = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="input_data")
    y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="output_data")

    # 3. 定义神经网络中间层（隐藏层）
    weights_L1 = tf.Variable(tf.random_normal(shape=[1, 10]))  # 这一层网络的权重矩阵
    biases_L1 = tf.Variable(tf.zeros(shape=[1, 10]))  # 这一层网络的偏置项
    result_L1 = tf.matmul(x, weights_L1) + biases_L1  # 中间层结果 = y * weight + biases
    L1_output = tf.nn.tanh(result_L1)  # 中间层输出 = 计算结果使用双曲正切函数（tanh）激活

    # 4. 定义神经网络输出层
    weights_L2 = tf.Variable(tf.random_normal(shape=[10, 1]))
    biases_L2 = tf.Variable(tf.zeros(shape=[1, 1]))
    result_L2 = tf.matmul(L1_output, weights_L2) + biases_L2
    prediction = tf.nn.tanh(result_L2)  # 预测操作

    # 5.定义损失函数与反向传播算法
    loss = tf.reduce_mean(tf.square(y - prediction))  # 损失函数
    optimizer = tf.train.GradientDescentOptimizer(0.1)  # 定义一个梯度下降优化器，学习率=0.1
    train_op = optimizer.minimize(loss)  # 训练操作，将优化器应用于损失函数，以最小化损失并更新参数

    # 添加 summary
    tf.summary.scalar("Loss", loss)  # 将损失记录到summary中
    merged_summary = tf.summary.merge_all()  # 合并所有的summary操作

    with tf.Session() as sess:
        # 创建summary 写入器
        writer = tf.summary.FileWriter(r"E:\ProgramFiles\CodeEnvironment\venv\logs", sess.graph)

        # 初始化变量
        init_op = tf.global_variables_initializer()
        sess.run(init_op)  # 运行初始化操作

        # 训练数据
        for i in range(2000):  # 迭代2000次
            _, summary = sess.run([train_op, merged_summary], feed_dict={x: x_data, y: y_data})
            writer.add_summary(summary, i)  # 将summary写入到日志文件中
        # 获得预测值
        prediction_value = sess.run(prediction, feed_dict={x: x_data})

        # 绘图
        plt.figure()
        plt.scatter(x_data, y_data)  # 散点为真实值
        plt.plot(x_data, prediction_value, "r-", lw=3)
        plt.show()

    # 关闭summary写入器
    writer.close()


if __name__ == '__main__':
    tensorflow_1()
    # tf_feed([3], [7])
    # tf_fetch()
    # tf_graph()
    # tf_graph_cpu()
    # tf_tensorboard()
    # tf_variable()