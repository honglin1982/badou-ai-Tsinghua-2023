import tensorflow.contrib.slim as slim
import tensorflow as tf



def vgg_16(inputs):

    with tf.variable_scope('vgg_16', 'vgg_16', [inputs]):

        # 构建vgg_16网络  输入为224*224*3  224是卷积默认的最佳尺寸
        # 两次3*3卷积 输出为64 slim.conv2d 步长默认1,1 padding默认same activation默认relu
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        # 2*2池化 输出为 112,112,64
        net = slim.max_pool2d(net, [2, 2], scope='pool1')

        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        # 2*2池化 输出为 56,56,128
        net = slim.max_pool2d(net, [2, 2], scope='pool2')

        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        # 2*2池化 输出为 28,28,256
        net = slim.max_pool2d(net, [2, 2], scope='pool3')

        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        # 2*2池化 输出为 14,14,512
        net = slim.max_pool2d(net, [2, 2], scope='pool4')

        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        # 2*2池化 输出为 7,7,512
        net = slim.max_pool2d(net, [2, 2], scope='pool5')

        # 利用卷积的方式模拟全连接层，效果等同，输出net为(1,1,1,4096)
        net = slim.conv2d(net, 4096, kernel_size=[7, 7], padding='valid', scope='fc6')
        net = slim.dropout(net, 0.5, is_training=True, scope='dropout6')
        # 利用卷积的方式模拟全连接层，效果等同，输出net为(1,1,1,4096)
        net = slim.conv2d(net, 4096, kernel_size=[1, 1], scope='fc7')
        net = slim.dropout(net, 0.5, is_training=True, scope='dropout7')
        # 利用卷积的方式模拟全连接层，效果等同，输出net为(1,1,1,1000)
        net = slim.conv2d(net, 1000, [1, 1], activation_fn=None, normalizer_fn=None, scope='fc8')

        # 由于用卷积的方式模拟全连接层，所以输出需要平铺 squeeze删除某个维度，删除的维度必须为1
        net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        return net

