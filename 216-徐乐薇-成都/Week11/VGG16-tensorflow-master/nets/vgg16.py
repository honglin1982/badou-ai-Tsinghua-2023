#-------------------------------------------------------------#
#   vgg16的网络部分
#-------------------------------------------------------------#
import tensorflow as tf

import tf_slim as slim

# 创建slim对象
slim = tf.contrib.slim                                                          # slim是一个轻量级的tensorflow库,用于定义、训练和评估复杂模型

def vgg_16(inputs,                                                              # 输入的图片
           num_classes=1000,                                                    # 分类的类别
           is_training=True,                                                    # 是否训练
           dropout_keep_prob=0.5,                                               # dropout的比例
           spatial_squeeze=True,                                                # 是否对输出进行squeeze操作,即去除维数为1的维度,如[1,1,3,4]输出[3,4]
           scope='vgg_16'):                                                     # scope的名字,默认vgg_16

    with tf.variable_scope(scope, 'vgg_16', [inputs]):                          # variable_scope的作用是指定共享变量的命名空间,scope为vgg_16,input为输入的图片
        # 建立vgg_16的网络

        # conv1两次[3,3]卷积网络，输出的特征层为64，输出为(224,224,64)
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')    # slim.repeat(input, n, op, *args, **kwargs)的作用是重复op操作n次
        # 2X2最大池化，输出net为(112,112,64)
        net = slim.max_pool2d(net, [2, 2], scope='pool1')                       # slim.max_pool2d(input, kernel_size, stride, padding, scope)的作用是最大池化操作

        # conv2两次[3,3]卷积网络，输出的特征层为128，输出net为(112,112,128)
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        # 2X2最大池化，输出net为(56,56,128)
        net = slim.max_pool2d(net, [2, 2], scope='pool2')

        # conv3三次[3,3]卷积网络，输出的特征层为256，输出net为(56,56,256)
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        # 2X2最大池化，输出net为(28,28,256)
        net = slim.max_pool2d(net, [2, 2], scope='pool3')

        # conv3三次[3,3]卷积网络，输出的特征层为256，输出net为(28,28,512)
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        # 2X2最大池化，输出net为(14,14,512)
        net = slim.max_pool2d(net, [2, 2], scope='pool4')

        # conv3三次[3,3]卷积网络，输出的特征层为256，输出net为(14,14,512)
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        # 2X2最大池化，输出net为(7,7,512)
        net = slim.max_pool2d(net, [2, 2], scope='pool5')

        # 利用卷积的方式模拟全连接层，效果等同，输出net为(1,1,4096)
        net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')      # slim.conv2d(input, num_outputs, kernel_size, stride, padding, scope)的作用是卷积操作
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,     # slim.dropout(input, keep_prob, is_training, scope)的作用是dropout操作,net为(1,1,4096),dropout_keep_prob为0.5,is_training为True
                            scope='dropout6')
        # 利用卷积的方式模拟全连接层，效果等同，输出net为(1,1,4096)
        net = slim.conv2d(net, 4096, [1, 1], scope='fc7')                       # slim.conv2d(input, num_outputs, kernel_size, stride, padding, scope)的作用是卷积操作
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,     # slim.dropout(input, keep_prob, is_training, scope)的作用是dropout操作,net为(1,1,4096),dropout_keep_prob为0.5,is_training为True
                            scope='dropout7')
        # 利用卷积的方式模拟全连接层，效果等同，输出net为(1,1,1000)
        net = slim.conv2d(net, num_classes, [1, 1],                             # slim.conv2d(input, num_outputs, kernel_size, stride, padding, scope)的作用是卷积操作
                        activation_fn=None,                                     # activation_fn为激活函数,这里为None
                        normalizer_fn=None,                                     # normalizer_fn为正则化函数,这里为None
                        scope='fc8')
        
        # 由于用卷积的方式模拟全连接层，所以输出需要平铺
        if spatial_squeeze:                                                     # spatial_squeeze为True
            net = tf.squeeze(net, [1, 2], name='fc8/squeezed')                  # tf.squeeze(input, axis=None, name=None, squeeze_dims=None)的作用是去除维数为1的维度,如[1,1,3,4]输出[3,4]
        return net