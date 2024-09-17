import math
import time
import numpy as np
import tensorflow as tf
import os
import glob
# tf.compat.v1.enable_eager_execution()  # 启用 Eager Execution
# tf.compat.v1.disable_eager_execution()  # 禁用 eager execution

# 训练集数据预处理
def preprocess_train_image(image, label):
    # 对图像进行预处理，并添加数据增强操作
    label = tf.cast(label, tf.int32)
    image = tf.cast(image, tf.float32)
    image = tf.random_crop(image, size=[24,24,3])  # 随机剪切
    image = tf.image.random_flip_left_right(image=image)  # 左右翻转
    image = tf.image.random_brightness(image=image, max_delta=0.8)  # 随机亮度，最大0.8
    image = tf.image.random_contrast(image=image, lower=0.2, upper=1.8)  # 随机对比度并指定上下界
    image = tf.image.per_image_standardization(image=image)  # 图像像素级标准化

    return image, label

# 测试集数据预处理
def preprocess_test_image(image, label):
    image = tf.image.resize_image_with_crop_or_pad(image=image,target_height=24,target_width=24)
    image = tf.image.per_image_standardization(image=image)

    return image, label

# 数据集读取函数
def read_cifar10(filename):
    record_bytes = 1 + 32 * 32 * 3  # 1 字节标签 + 32x32x3 字节图像
    dataset = tf.data.FixedLengthRecordDataset(filename, record_bytes)  # 该类用于从一个包含固定长度记录的文件中读取数据

    def _parse_function(example):
        # 解析每条记录
        record = tf.io.decode_raw(example, tf.uint8)  # 需要注意类型相同
        label = tf.cast(record[0], tf.int32)  # 每条记录的第一个字符为标签
        image = tf.reshape(record[1:], (3, 32, 32))  # 转换为 (channels, height, width)
        image = tf.transpose(image, perm=(1, 2, 0))  # 转换为 (height, width, channels)
        return image, label

    dataset = dataset.map(_parse_function)  # 调用内函数进行处理
    return dataset

# 创建带有权重正则化损失的神经网络权重变量，并将相应的损失添加到损失集合中，以便在计算总体损失时一并考虑
def variable_with_weight_loss(shape, stddev, w1=None):
    """
        1.shape 指定权重的形状（即卷积核大小）
        2.stddev 指定正态分布的标准差
        3.w1 控制L2 loss的大小
        4.tf.nn.l2_loss()计算权重L2 loss
        3.使用函数tf.multiply()计算权重L2 loss与w1的乘积，并赋值给weights_loss
        4.使用函数tf.add_to_collection()将最终的结果放在名为losses的集合里面，方便后面计算神经网络的总体loss，
    """
    normal_tensor = tf.truncated_normal(shape=shape, stddev=stddev)  # 生成截断正态分布的随机数张量
    weight_var = tf.Variable(normal_tensor)  # 该张量=神经网络的权重参数
    if w1 is not None:  # 判断是否对权重参数应用 L2 正则化
        L2_square = tf.nn.l2_loss(weight_var)  # 计算 L2 loss
        # 加权的L2损失，可以在计算训练时的总损失时与其他损失相加
        weights_loss = tf.multiply(L2_square, w1, name="weights_loss")  # 进行张量逐元素乘，并进行操作名称设定
        tf.add_to_collection("losses", weights_loss)  # 该函数用与将某个对象添加至集合中
    return weight_var  # 返回权重参数


def tf_cifar10():
    # 因为后面定义全连接网络时用到了batch_size，所以x中第一个参数不应该是None，而应该是batch_size
    x = tf.placeholder(tf.float32, shape=[batch_size, 24, 24, 3])  # 创建占位符节点x、y
    y = tf.placeholder(tf.int32, shape=[batch_size])

    # 创建第一个卷积层 shape=(kh,kw,ci,co)  ci 输入RGB 3通道  co 输出64通道
    kernel_1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, w1=0.0)  # 5e-2=0.05
    conv_1 = tf.nn.conv2d(x, filters=kernel_1, strides=[1, 1, 1, 1], padding="SAME")  # SAME 在输入周围进行0填充，使输入与输出的空间维度相同
    bias_1 = tf.Variable(tf.constant(0.0, shape=[64]))  # 创建变量：为一个常量张量，值不变。用于偏置项
    relu_1 = tf.nn.relu(tf.nn.bias_add(conv_1, bias_1))  # 应用ReLU激活函数：并将卷积的结果添加偏置
    pool_1 = tf.nn.max_pool(relu_1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")  # 执行最大池化操作
    # 创建第二个卷积层
    kernel_2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, w1=0.0)
    conv_2 = tf.nn.conv2d(pool_1, filters=kernel_2, strides=[1, 1, 1, 1], padding="SAME")
    bias_2 = tf.Variable(tf.constant(0.1, shape=[64]))
    relu_2 = tf.nn.relu(tf.nn.bias_add(conv_2, bias_2))
    pool_2 = tf.nn.max_pool(relu_2, ksize=[1,3,3,1], strides=[1,2,2,1], padding="SAME")

    # 全连接操作前需要将 pool_2 的输出一维化
    unidimensional = tf.reshape(pool_2, shape=[batch_size, -1])  # -1 代表将 pool_2的h,w,c三维结构拉直为一维结构
    dim = unidimensional.get_shape()[1].value  # 即获取一维化后的数据大小

    # 建立第一个全连接层: 输入通道数为一维化的 dim， 输出通道为384
    weight_1 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, w1=0.004)
    fc_bias_1 = tf.Variable(tf.constant(0.1, shape=[384]))
    fc_1 = tf.nn.relu(tf.matmul(unidimensional, weight_1) + fc_bias_1)
    # 建立第二个全连接层：输入通道384，输出192
    weight_2 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, w1=0.004)
    fc_bias_2 = tf.Variable(tf.constant(0.1, shape=[192]))
    local_4 = tf.nn.relu(tf.matmul(fc_1, weight_2) + fc_bias_2)
    # 建立第三个全连接层：输入192，输出10
    weight_3 = variable_with_weight_loss(shape=[192,10], stddev=1/192.0, w1=0.0)
    fc_bias_3 = tf.Variable(tf.constant(0.1, shape=[10]))
    result_logits = tf.add(tf.matmul(local_4, weight_3), fc_bias_3)

    # 计算损失：包括权重参数的正则化损失与交叉熵损失
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result_logits, labels=tf.cast(y,tf.int64))
    weight_with_l2_loss = tf.add_n(tf.get_collection("losses"))  # 将集合 losses 中的所有元素相加得到L2正则化损失
    loss = tf.reduce_mean(cross_entropy) + weight_with_l2_loss  # 总损失
    # 创建Adam优化器，设置学习率为 0.001，并最小化损失
    train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

    # 输出结果中top k的准确率，k=1时 输出分类准确率最高时的数值
    top_k_op = tf.nn.in_top_k(predictions=result_logits, targets=y, k=1)
    # 创建一个可初始化的迭代器
    train_iterator = train_dataset.make_initializable_iterator()
    test_iterator = test_dataset.make_initializable_iterator()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # 初始化全局变量
        sess.run(train_iterator.initializer)  # 初始化迭代器
        sess.run(test_iterator.initializer)  # 初始化迭代器
        # 从数据集中获取下一个批次数据元素，通过变量在循环中进行调用覆盖原有数据
        next_train_element = train_iterator.get_next()
        next_test_element = test_iterator.get_next()

        for step in range(1, max_steps+1):
            start_time = time.time()
            image_batch, label_batch = sess.run(next_train_element)  # 从训练队列中获取一个 batch 的训练数据
            _, loss_value = sess.run(fetches=[train_op, loss], feed_dict={x:image_batch, y:label_batch})

            duration = time.time() - start_time  # 花费时间
            # 每隔100step会计算展示当前loss、每秒训练样本数、训练一个batch花费的时间
            if step % 100 == 0:
                examples_per_sec = batch_size / duration  # 每秒钟训练的样本数量
                sec_per_batch = float(duration)  # 每个 batch 的训练时间。
                print("step %d, loss=%.2f(examples/sec=%.1f ;sec/batch=%.3f)"
                      % (step, loss_value, examples_per_sec, sec_per_batch))

        print()
        # 在测试集上进行评估
        number_batch = math.ceil(test_set_number / batch_size)  # 计算测试数据集需要被分成多少个测试批次,math.ceil向上取整
        true_count = 0  # 统计正确预测的样本数量

        for j in range(number_batch):  # 迭代计算每个测试批次的预测结果
            image_batch, label_batch = sess.run(next_test_element)
            predictions = sess.run(fetches=[top_k_op], feed_dict={x:image_batch, y:label_batch})
            true_count += np.sum(predictions)  # 将当前测试批次中正确预测的样本数量累加到 true_count 中
        print("accuracy = %.3f%%" % ((true_count/test_set_number) * 100), true_count)


if __name__ == '__main__':
    # 参数设置
    batch_size = 100  # 批次大小
    cifar_data_dir = "cifar-10-batches-bin"  # 数据集目录
    max_steps = 5000
    test_set_number = 10000
    train_set_number = 50000

    # 创建训练集
    train_dataset = read_cifar10(glob.glob(os.path.join(cifar_data_dir, "data_batch_*.bin")))
    train_dataset = train_dataset.map(preprocess_train_image)  # map()为惰性操作，在数据集被迭代时才逐个应用函数返回处理后的元素
    # 随机10000,按批次,预取3,重复10次，50w数据
    train_dataset = train_dataset.shuffle(buffer_size=10000).batch(batch_size).prefetch(buffer_size=3).repeat(count=10)

    # 创建测试集
    test_dataset = read_cifar10(os.path.join(cifar_data_dir, "test_batch.bin"))  # 为一个能够逐个提供图像-标签对的迭代对象
    test_dataset = test_dataset.map(preprocess_test_image)
    test_dataset = test_dataset.batch(batch_size).prefetch(buffer_size=3)  # 不随机而是顺序执行

    tf_cifar10()
