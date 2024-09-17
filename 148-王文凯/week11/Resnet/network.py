import tensorflow as tf


def resnet_50(inputs, output_shape, is_training):
    zero_pad = tf.pad(inputs, paddings=[[0, 0], [3, 3], [3, 3], [0, 0]], mode='CONSTANT', constant_values=0)
    conv_1 = tf.layers.conv2d(
        inputs=zero_pad,
        filters=64,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding='VALID',
        activation=None
    )
    bn_1 = tf.layers.batch_normalization(inputs=conv_1, training=is_training)
    relu_1 = tf.nn.relu(bn_1)
    pool_1 = tf.layers.max_pooling2d(inputs=relu_1, pool_size=(3, 3), strides=(2, 2))

    filters = [64, 64, 256]
    kernel_size = (3, 3)
    conv_block = generate_conv_block(
        inputs=pool_1,
        kernel_size=kernel_size,
        filters=filters,
        training=is_training,
        strides=(1, 1)
    )
    identity_block = generate_identity_block(
        inputs=conv_block,
        kernel_size=kernel_size,
        filters=filters,
        training=is_training
    )
    identity_block = generate_identity_block(
        inputs=identity_block,
        kernel_size=kernel_size,
        filters=filters,
        training=is_training
    )

    filters = [128, 128, 512]
    conv_block = generate_conv_block(
        inputs=identity_block,
        kernel_size=kernel_size,
        filters=filters,
        training=is_training,
    )
    identity_block = generate_identity_block(
        inputs=conv_block,
        kernel_size=kernel_size,
        filters=filters,
        training=is_training
    )
    identity_block = generate_identity_block(
        inputs=identity_block,
        kernel_size=kernel_size,
        filters=filters,
        training=is_training
    )
    identity_block = generate_identity_block(
        inputs=identity_block,
        kernel_size=kernel_size,
        filters=filters,
        training=is_training
    )

    filters = [256, 256, 1024]
    conv_block = generate_conv_block(
        inputs=identity_block,
        kernel_size=kernel_size,
        filters=filters,
        training=is_training,
    )
    identity_block = generate_identity_block(
        inputs=conv_block,
        kernel_size=kernel_size,
        filters=filters,
        training=is_training
    )
    identity_block = generate_identity_block(
        inputs=identity_block,
        kernel_size=kernel_size,
        filters=filters,
        training=is_training
    )
    identity_block = generate_identity_block(
        inputs=identity_block,
        kernel_size=kernel_size,
        filters=filters,
        training=is_training
    )
    identity_block = generate_identity_block(
        inputs=identity_block,
        kernel_size=kernel_size,
        filters=filters,
        training=is_training
    )
    identity_block = generate_identity_block(
        inputs=identity_block,
        kernel_size=kernel_size,
        filters=filters,
        training=is_training
    )

    filters = [512, 512, 2048]
    conv_block = generate_conv_block(
        inputs=identity_block,
        kernel_size=kernel_size,
        filters=filters,
        training=is_training,
    )
    identity_block = generate_identity_block(
        inputs=conv_block,
        kernel_size=kernel_size,
        filters=filters,
        training=is_training
    )
    identity_block = generate_identity_block(
        inputs=identity_block,
        kernel_size=kernel_size,
        filters=filters,
        training=is_training
    )

    pool_2 = tf.layers.average_pooling2d(inputs=identity_block, pool_size=(7, 7), strides=(1, 1))
    flatten_1 = tf.layers.flatten(pool_2)

    outputs = tf.layers.dense(
        inputs=flatten_1,
        units=output_shape,
        activation=tf.nn.softmax
    )

    return outputs


def generate_identity_block(inputs, kernel_size, filters, training):
    filters_1, filters_2, filters_3 = filters

    conv_1 = tf.layers.conv2d(
        inputs=inputs,
        kernel_size=(1, 1),
        filters=filters_1,
        padding='SAME',
        activation=None
    )
    bn_1 = tf.layers.batch_normalization(inputs=conv_1, training=training)
    relu_1 = tf.nn.relu(bn_1)

    conv_2 = tf.layers.conv2d(
        inputs=relu_1,
        kernel_size=kernel_size,
        filters=filters_2,
        padding='SAME',
        activation=None
    )
    bn_2 = tf.layers.batch_normalization(inputs=conv_2, training=training)
    relu_2 = tf.nn.relu(bn_2)

    conv_3 = tf.layers.conv2d(
        inputs=relu_2,
        kernel_size=(1, 1),
        filters=filters_3,
        padding='SAME',
        activation=None
    )
    bn_2 = tf.layers.batch_normalization(inputs=conv_3, training=training)
    # print('bn_identity', bn_2.shape)
    # print('inputs', inputs.shape)

    outputs = tf.nn.relu(tf.add(bn_2, inputs))

    return outputs


def generate_conv_block(inputs, kernel_size, filters, training, strides=(2, 2)):
    filters_1, filters_2, filters_3 = filters

    conv_1 = tf.layers.conv2d(
        inputs=inputs,
        kernel_size=(1, 1),
        filters=filters_1,
        strides=strides,
        padding='VALID',
        activation=None
    )
    bn_1 = tf.layers.batch_normalization(inputs=conv_1, training=training)
    relu_1 = tf.nn.relu(bn_1)

    conv_2 = tf.layers.conv2d(
        inputs=relu_1,
        kernel_size=kernel_size,
        filters=filters_2,
        padding='SAME',
        activation=None
    )
    bn_2 = tf.layers.batch_normalization(inputs=conv_2, training=training)
    relu_2 = tf.nn.relu(bn_2)

    conv_3 = tf.layers.conv2d(
        inputs=relu_2,
        kernel_size=(1, 1),
        filters=filters_3,
        padding='SAME',
        activation=None
    )
    bn_3 = tf.layers.batch_normalization(inputs=conv_3, training=training)

    shortcut = tf.layers.conv2d(
        inputs=inputs,
        kernel_size=(1, 1),
        filters=filters_3,
        strides=strides,
        padding='SAME',
        activation=None
    )
    bn_shortcut = tf.layers.batch_normalization(inputs=shortcut, training=training)

    # print('bn_conv', bn_3.shape)
    # print('shortcut', bn_shortcut.shape)
    outputs = tf.nn.relu(tf.add(bn_3, bn_shortcut))

    return outputs


