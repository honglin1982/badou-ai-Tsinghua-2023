import tensorflow as tf


class Cifar10Model:

    @staticmethod
    def weight_loss(shape, stddev, w1):
        """
        stddev: 标准差 eg: 5e-2 表示 0.05
        """
        # tf.truncated_normal 随机生成张量
        var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
        if w1 is not None:
            # 求带权重的损失函数
            # tf.nn.l2_loss L2 损失函数，张量中所有元素的平方和
            # 公式：Σ(var_i^2)
            weights_loss = tf.multiply(tf.nn.l2_loss(var), w1, name="weights_loss")
            tf.add_to_collection("losses", weights_loss)
        return var

    def build(self, input_x, input_y, batch):
        print("gen")
        # 卷积层
        kernel1 = self.weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, w1=0.0)
        conv1 = tf.nn.conv2d(input_x, kernel1, strides=[1, 1, 1, 1], padding="SAME")
        bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, bias1))
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

        kernel2 = self.weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, w1=0.0)
        conv2 = tf.nn.conv2d(pool1, kernel2, strides=[1, 1, 1, 1], padding="SAME")
        bias2 = tf.Variable(tf.constant(0.0, shape=[64]))
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, bias2))
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

        # 全连接层
        reshape = tf.reshape(pool2, [batch, -1])
        dim = reshape.get_shape()[1].value

        fc_w1 = self.weight_loss(shape=[dim, 384], stddev=0.04, w1=0.004)
        fc_b1 = tf.Variable(tf.constant(0.1, shape=[384]))
        fc_1 = tf.nn.relu(tf.matmul(reshape, fc_w1) + fc_b1)

        fc_w2 = self.weight_loss(shape=[384, 192], stddev=0.04, w1=0.004)
        fc_b2 = tf.Variable(tf.constant(0.1, shape=[192]))
        fc_2 = tf.nn.relu(tf.matmul(fc_1, fc_w2) + fc_b2)

        fc_w3 = self.weight_loss(shape=[192, 10], stddev=0.04, w1=0.004)
        fc_b3 = tf.Variable(tf.constant(0.1, shape=[10]))
        fc_3 = tf.add(tf.matmul(fc_2, fc_w3), fc_b3)

        # 计算损失，包括权重参数的正则化损失和交叉熵损失
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fc_3, labels=tf.cast(input_y, tf.int64))

        weights_with_l2_loss = tf.add_n(tf.get_collection("losses"))
        loss = tf.reduce_mean(cross_entropy) + weights_with_l2_loss

        train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
        return train_op, fc_3, loss
