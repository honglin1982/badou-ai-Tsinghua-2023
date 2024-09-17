import tensorflow as tf


def vgg_19(inputs, output_shape):
    # conv_1 2
    conv_1 = tf.layers.conv2d(
        inputs,
        filters=64,
        kernel_size=[3, 3],
        padding='SAME',
        activation=tf.nn.relu
    )
    conv_1 = tf.layers.conv2d(
        conv_1,
        filters=64,
        kernel_size=[3, 3],
        padding='SAME',
        activation=tf.nn.relu
    )
    # print('conv_1', conv_1.shape)
    pool_1 = tf.layers.max_pooling2d(
        conv_1,
        pool_size=[2, 2],
        strides=(2, 2),
        padding='VALID'
    )
    # print('pool_1', pool_1.shape)

    # conv_2 2
    conv_2 = tf.layers.conv2d(
        pool_1,
        filters=128,
        kernel_size=[3, 3],
        padding='SAME',
        activation=tf.nn.relu
    )
    conv_2 = tf.layers.conv2d(
        conv_2,
        filters=128,
        kernel_size=[3, 3],
        padding='SAME',
        activation=tf.nn.relu
    )
    pool_2 = tf.layers.max_pooling2d(
        conv_2,
        pool_size=[2, 2],
        strides=(2, 2),
        padding='VALID'
    )
    # print('pool_2', pool_2.shape)

    # conv_3 4
    conv_3 = tf.layers.conv2d(
        pool_2,
        filters=256,
        kernel_size=[3, 3],
        padding='SAME',
        activation=tf.nn.relu
    )
    # conv_3 = tf.layers.conv2d(
    #     conv_3,
    #     filters=256,
    #     kernel_size=[3, 3],
    #     padding='SAME',
    #     activation=tf.nn.relu
    # )
    conv_3 = tf.layers.conv2d(
        conv_3,
        filters=256,
        kernel_size=[3, 3],
        padding='SAME',
        activation=tf.nn.relu
    )
    conv_3 = tf.layers.conv2d(
        conv_3,
        filters=256,
        kernel_size=[3, 3],
        padding='SAME',
        activation=tf.nn.relu
    )
    pool_3 = tf.layers.max_pooling2d(
        conv_3,
        pool_size=[2, 2],
        strides=(2, 2),
        padding='VALID'
    )
    # print('pool_3', pool_3.shape)

    # conv_4 4
    conv_4 = tf.layers.conv2d(
        pool_3,
        filters=512,
        kernel_size=[3, 3],
        padding='SAME',
        activation=tf.nn.relu
    )
    # conv_4 = tf.layers.conv2d(
    #     conv_4,
    #     filters=512,
    #     kernel_size=[3, 3],
    #     padding='SAME',
    #     activation=tf.nn.relu
    # )
    conv_4 = tf.layers.conv2d(
        conv_4,
        filters=512,
        kernel_size=[3, 3],
        padding='SAME',
        activation=tf.nn.relu
    )
    conv_4 = tf.layers.conv2d(
        conv_4,
        filters=512,
        kernel_size=[3, 3],
        padding='SAME',
        activation=tf.nn.relu
    )
    pool_4 = tf.layers.max_pooling2d(
        conv_4,
        pool_size=[2, 2],
        strides=(2, 2),
        padding='VALID'
    )
    # print('pool_4', pool_4.shape)

    # conv_5 4
    conv_5 = tf.layers.conv2d(
        pool_4,
        filters=512,
        kernel_size=[3, 3],
        padding='SAME',
        activation=tf.nn.relu
    )
    # conv_5 = tf.layers.conv2d(
    #     conv_5,
    #     filters=512,
    #     kernel_size=[3, 3],
    #     padding='SAME',
    #     activation=tf.nn.relu
    # )
    conv_5 = tf.layers.conv2d(
        conv_5,
        filters=512,
        kernel_size=[3, 3],
        padding='SAME',
        activation=tf.nn.relu
    )
    conv_5 = tf.layers.conv2d(
        conv_5,
        filters=512,
        kernel_size=[3, 3],
        padding='SAME',
        activation=tf.nn.relu
    )
    pool_5 = tf.layers.max_pooling2d(
        conv_5,
        pool_size=[2, 2],
        strides=(2, 2),
        padding='VALID'
    )
    # print('pool_5', pool_5.shape)

    # fc 3 优化 卷积层模拟替代全连接层，去除矩阵拉平操作
    fc_1 = tf.layers.conv2d(
        pool_5,
        filters=4096,
        kernel_size=[7, 7],
        padding='VALID',
        activation=tf.nn.relu
    )
    fc_1 = tf.nn.dropout(fc_1, 0.5)
    # print('fc_1', fc_1.shape)

    fc_2 = tf.layers.conv2d(
        fc_1,
        filters=4096,
        kernel_size=[1, 1],
        padding='SAME',
        activation=tf.nn.relu
    )
    fc_2 = tf.nn.dropout(fc_2, 0.5)

    output_layer = tf.layers.conv2d(
        fc_2,
        filters=output_shape,
        kernel_size=[1, 1],
        padding='VALID',
        activation=None
    )

    output_layer = tf.squeeze(output_layer, [1, 2], name='output_layer_squeezed')
    # print('output_layer', output_layer.shape)

    return output_layer





