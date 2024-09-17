import tensorflow as tf

slim = tf.contrib.slim


def vgg_16(inputs, num_classes=1000, is_training=True, dropout_keep_prob=0.5, spatial_squeeze=True, scope='vgg_16'):
    with tf.variable_scope(scope, 'vgg_16', [inputs]):
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')

        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
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

        net = slim.conv2d(net, 4096, [7, 7], padding='valid', scope='fc6')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout6')

        net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout7')

        net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, nolmalizer_fn=None, scope='fc8')

        if spatial_squeeze:
            net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        return net
