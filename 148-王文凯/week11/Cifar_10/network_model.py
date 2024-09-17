import tensorflow as tf
import time


class ConvolutionNeuralNetworkForCifar:
    def __init__(self, height, width, channels, batch_size):
        self.x = tf.placeholder(tf.float32, [batch_size, height, width, channels])
        self.y_ = tf.placeholder(tf.int32, [batch_size])

        # conv_1 shape = [kh, kw, ci, co]
        self.kernel_1 = self.variable_with_weight_loss(shape=[5, 5, channels, 64], stddev=5e-2, w=0.0)
        self.conv_1 = tf.nn.conv2d(self.x, self.kernel_1, [1, 1, 1, 1], padding="SAME")
        self.bias_1 = tf.Variable(tf.constant(0.0, shape=[64]))
        self.relu_1 = tf.nn.relu(tf.nn.bias_add(self.conv_1, self.bias_1))
        self.pool_1 = tf.nn.max_pool(self.relu_1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

        # conv_2
        self.kernel_2 = self.variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, w=0.0)
        self.conv_2 = tf.nn.conv2d(self.pool_1, self.kernel_2, [1, 1, 1, 1], padding="SAME")
        self.bias_2 = tf.Variable(tf.constant(0.1, shape=[64]))
        self.relu_2 = tf.nn.relu(tf.nn.bias_add(self.conv_2, self.bias_2))
        self.pool_2 = tf.nn.max_pool(self.relu_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

        # 池化层输出扁平化为一维向量
        self.reshape = tf.reshape(self.pool_2, [batch_size, -1])
        self.dim = self.reshape.get_shape()[1].value

        # full_connected_layer_1
        self.fc_1_weight = self.variable_with_weight_loss(shape=[self.dim, 512], stddev=5e-2, w=0.01)
        self.fc_1_bias = tf.Variable(tf.constant(0.1, shape=[512]))
        self.fc_1 = tf.nn.relu(tf.matmul(self.reshape, self.fc_1_weight) + self.fc_1_bias)

        # fc_2
        self.fc_2_weight = self.variable_with_weight_loss(shape=[512, 256], stddev=0.04, w=0.01)
        self.fc_2_bias = tf.Variable(tf.constant(0.1, shape=[256]))
        self.fc_2 = tf.nn.relu(tf.matmul(self.fc_1, self.fc_2_weight) + self.fc_2_bias)

        # fc_3
        self.fc_3_weight = self.variable_with_weight_loss(shape=[256, 10], stddev=1 / 256.0, w=0.0)
        self.fc_3_bias = tf.Variable(tf.constant(0.1, shape=[10]))
        self.res = tf.add(tf.matmul(self.fc_2, self.fc_3_weight), self.fc_3_bias)

        self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.res,
                                                                            labels=tf.cast(self.y_, tf.int64))
        self.weights_with_l2_loss = tf.add_n(tf.get_collection("losses"))
        self.loss = tf.reduce_mean(self.cross_entropy) + self.weights_with_l2_loss

        self.train_op = tf.train.AdamOptimizer(1e-3).minimize(self.loss)
        self.top_k_op = tf.nn.in_top_k(self.res, self.y_, 1)
        self.init_op = tf.global_variables_initializer()

    @staticmethod
    def variable_with_weight_loss(shape, stddev, w):
        var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
        if w:
            w_loss = tf.multiply(tf.nn.l2_loss(var), w, name="weights_loss")
            tf.add_to_collection("losses", w_loss)
        return var

