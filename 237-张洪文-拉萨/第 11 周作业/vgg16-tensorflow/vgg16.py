import tensorflow as tf

# vgg16网络模型

# 创建slim对象:用于定义、训练和评估神经网络模型
slim = tf.contrib.slim

def vgg_16(inputs, num_classes=1000, is_training=True,
           dropout_keep_prob=0.5, spatial_squeeze=True, scope="vgg_16"):
    # 创建命名空间：指定名称、变量前缀、
    with tf.variable_scope(scope, "vgg_16", [inputs]):
        # 开始建立 vgg_16 网络

        # conv1 2次 3*3 卷积，输出特征 64， 输出（224,224,64）, 变量作用域 conv1
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3,3], scope="conv1")
        # 2*2最大池化，输出（112,112,64）
        net = slim.max_pool2d(net, [2,2], scope="pool1")

        # conv2 2次3*3卷积，输出特征128，输出shape（112,112,128）
        net = slim.repeat(net, 2, slim.conv2d, 128, [3,3], scope="conv2")
        # 2*2最大池化，输出（56,56,128）
        net = slim.max_pool2d(net, [2,2],scope="pool2")

        # conv3 3次3*3卷积，输出特征256，输出shape（56,56,256）
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope="conv3")
        # 2*2最大池化，输出（28,28,256）
        net = slim.max_pool2d(net, [2, 2], scope="pool3")

        # conv4 3次3*3卷积，输出特征512，输出shape（28,28,512）
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope="conv4")
        # 2*2最大池化，输出（14,14,512）
        net = slim.max_pool2d(net, [2, 2], scope="pool4")

        # conv5 3次3*3卷积，输出特征512，输出shape（14,14,512）
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope="conv5")
        # 2*2最大池化，输出（7,7,512）
        net = slim.max_pool2d(net, [2, 2], scope="pool5")

        # fc6 卷积模拟全连接层，输出（1,1,4096）
        net = slim.conv2d(net, 4096, [7,7], padding="VALID", scope="fc6")  # 不填充
        # 丢弃层: 减轻过拟合，每个神经元保留的概率为 dropout_keep_prob
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope="dropout6")

        # fc7 卷积模拟全连接层，输出（1,1,4096）
        net = slim.conv2d(net, 4096, [1, 1], scope="fc7")  # 0填充
        # 丢弃层: 减轻过拟合，每个神经元保留的概率为 dropout_keep_prob
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope="dropout7")

        # fc8 最后的分类
        net = slim.conv2d(net, num_classes, [1,1], activation_fn=None, normalizer_fn=None, scope="fc8")

        if spatial_squeeze:
            # 删除 1，2 维度位置为1的维度
            net = tf.squeeze(net, axis=[1,2], name="fc8/squeezed")

        return net

