#vgg 网络搭建
import tensorflow as tf

#创建slim对象 保证 tensorflow 和 tensorflow.estimator 大版本号一致
slim = tf.contrib.slim

def vgg_16(inputs, numClasses=1000, isTraining=True, dropoutKeepProb=0.5,
           spatialSqueeze=True, scope='vgg_16'):

    with tf.variable_scope(scope, 'vgg_16', [inputs]):

        #构建vgg16网络
        #搭建两层[3, 3]卷积网络，输出特征层64，输出为(224, 224, 64)
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        # 2 * 2 最大池化， 输出(112, 112, 64)
        net = slim.max_pool2d(net, [2, 2], scope='pool1')

        #再搭建两层[3, 3]卷积网络，输出特征层128，输出为(112, 112, 128)
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        # 2 * 2 最大池化，输出(56, 56, 128)
        net = slim.max_pool2d(net, [2, 2], scope='pool2')

        #再搭建三层[3,3]卷积网络，输出特征层256，输出为(56, 56, 256)
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        #2 * 2最大池化，输出net为(28, 28, 256)
        net = slim.max_pool2d(net, [2, 2], scope='pool3')

        #再搭建三层[3, 3]卷积网络，输出特征层512，输出(28, 28, 512)
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        net = slim.max_pool2d(net, [2, 2], scope='pool4') #输出(14, 14, 512)

        #输出(14, 14, 512)
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        net = slim.max_pool2d(net, [2, 2], scope='pool5') #输出(7, 7, 512)

        #利用卷积的方式模拟全连接层 输出(7, 7, 4096)
        net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
        net = slim.dropout(net, dropoutKeepProb, is_training=isTraining,
                           scope='dropout6')
        #输出(1, 1, 4096)
        net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
        net = slim.dropout(net, dropoutKeepProb,  is_training=isTraining,
                           scope='dropout7')

        #同样卷积模拟全连接，输出(1, 1, 1000)
        net = slim.conv2d(net, numClasses, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='fc8')
        #由于用卷积模拟全连接 平铺所有输出
        if spatialSqueeze: net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        return net;
