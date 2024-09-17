import tensorflow as tf

slim = tf.contrib.slim


def vgg_16(inputs, outputs, scope='vgg_16', is_training=True):
    with tf.variable_scope(scope, 'vgg_16', [inputs]):
        # conv_1 2
        conv_1 = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv_1')
        pool_1 = slim.max_pool2d(conv_1, [2, 2], scope='pool_1')

        # conv_2 3
        conv_2 = slim.repeat(pool_1, 3, slim.conv2d, 128, [3, 3], scope='conv_2')
        pool_2 = slim.max_pool2d(conv_2, [2, 2], scope='pool_2')

        # conv_3 3
        conv_3 = slim.repeat(pool_2, 3, slim.conv2d, 256, [3, 3], scope='conv_3')
        pool_3 = slim.max_pool2d(conv_3, [2, 2], scope='pool_3')

        # conv_4 3
        conv_4 = slim.repeat(pool_3, 3, slim.conv2d, 512, [3, 3], scope='conv_4')
        pool_4 = slim.max_pool2d(conv_4, [2, 2], scope='pool_4')

        # conv_5 3
        conv_5 = slim.repeat(pool_4, 3, slim.conv2d, 512, [3, 3], scope='conv_5')
        pool_5 = slim.max_pool2d(conv_5, [2, 2], scope='pool_5')

        # fc 3
        fc_1 = slim.conv2d(pool_5, 4096, [7, 7], padding='VALID', scope='fc_1')
        fc_1 = slim.dropout(fc_1, 0.5, is_training=is_training, scope='dropout_1')

        fc_2 = slim.conv2d(fc_1, 4096, [1, 1], scope='fc_2')
        fc_2 = slim.dropout(fc_2, 0.5, is_training=is_training, scope='dropout_2')

        fc_3 = slim.conv2d(fc_2, outputs, [1, 1], activation_fn=None, normalizer_fn=None, scope='fc3')
        fc_3 = tf.squeeze(fc_3, [1, 2], name='fc3_squeeze')

        return fc_3


