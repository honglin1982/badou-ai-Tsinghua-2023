# -------------------------------------------------------------#
#   vgg16的网络部分 tensorflow
# -------------------------------------------------------------#
import tensorflow as tf
'''
从 Tensorflow 2.0 开始，tf 移除了 tensorflow.contrib 库，其曾经 链接 也已失效，转而作为 tf_slim 模块单独下载使用。因此，在 tf2 版本中无法再使用 import tf.contrib.slim as slim 调用 slim 库。
下载：pip install --upgrade tf_slim
调用：import tf_slim as slim
关于 slim 下找不到 utils 的问题，请将 slim.utils 替换成 slim.layers.utils 即可。
'''
# 创建slim对象
# slim = tf.contrib.slim
import tf_slim as slim

tf.compat.v1.disable_eager_execution()


def vgg_16(inputs, num_classes=1000, is_training=True, dropout_keep_prob=0.5, spatial_squeeze=True, scope='vgg_16'):

    with tf.compat.v1.variable_scope(scope, 'vgg_16', [inputs]):
        # 建立vgg_16的网络

        # conv1两次[3,3]卷积网络，输出的特征层为64，输出为(224,224,64)
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        # 2X2最大池化，输出net为(112,112,64)
        net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool1')

        # conv2两次[3,3]卷积网络，输出的特征层为128，输出net为(112,112,128)
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        # 2X2最大池化，输出net为(56,56,128)
        net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool2')

        # conv3三次[3,3]卷积网络，输出的特征层为256，输出net为(56,56,256)
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        # 2X2最大池化，输出net为(28,28,256)
        net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool3')

        # conv3三次[3,3]卷积网络，输出的特征层为256，输出net为(28,28,512)
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        # 2X2最大池化，输出net为(14,14,512)
        net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool4')

        # conv3三次[3,3]卷积网络，输出的特征层为256，输出net为(14,14,512)
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        # 2X2最大池化，输出net为(7,7,512)
        net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool5')

        # 两次全连接卷积替代
        # 利用卷积的方式模拟全连接层，效果等同，输出net为(1,1,4096)
        net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout6')
        # 利用卷积的方式模拟全连接层，效果等同，输出net为(1,1,4096)
        net = slim.conv2d(net, 4096, [1, 1], padding='SAME', scope='fc7')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout7')

        # 利用卷积的方式模拟全连接层，效果等同，输出net为(1,1,1000)
        net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='fc8')

        # 由于用卷积的方式模拟全连接层，所以输出需要平铺
        if spatial_squeeze:
            # todo tf.squeeze,如果没有指定某个int数值维度，那就默认删除维度为1，如果指定了并且是一个list，那就根据数值的索引来删除
            net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        return net

