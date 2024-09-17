import tensorflow as tf
import time


class Mobilenet:
    def __init__(self, input_shape, output_shape, depth_multiplier=1):
        self.x = tf.placeholder(tf.float32, shape=[None] + input_shape)
        self.y = tf.placeholder(tf.int32, shape=[None, output_shape])
        self.training = tf.placeholder(tf.bool)

        # 224 * 224 * 3 -> 112 * 112 * 32
        conv_1 = self.conv_block(
            inputs=self.x,
            filters=32,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='SAME',
            training=self.training
        )

        # 112 * 112 * 32 -> 112 * 112 * 64
        conv_2 = self.depth_wise_conv_block(
            inputs=conv_1,
            filters=32,
            point_wise_conv_filters=64,
            depth_multiplier=depth_multiplier,
            strides=(1, 1),
            training=self.training
        )

        # 112 * 112 * 64 -> 56 * 56 * 128
        conv_3 = self.depth_wise_conv_block(
            inputs=conv_2,
            filters=64,
            point_wise_conv_filters=128,
            depth_multiplier=depth_multiplier,
            strides=(2, 2),
            training=self.training
        )

        # 56 * 56 * 128 -> 56 * 56 * 128
        conv_4 = self.depth_wise_conv_block(
            inputs=conv_3,
            filters=128,
            point_wise_conv_filters=128,
            depth_multiplier=depth_multiplier,
            strides=(1, 1),
            training=self.training
        )

        # 56 * 56 * 128 -> 28 * 28 * 256
        conv_5 = self.depth_wise_conv_block(
            inputs=conv_4,
            filters=128,
            point_wise_conv_filters=256,
            depth_multiplier=depth_multiplier,
            strides=(2, 2),
            training=self.training
        )

        # 28 * 28 * 256 -> 28 * 28 * 256
        conv_6 = self.depth_wise_conv_block(
            inputs=conv_5,
            filters=256,
            point_wise_conv_filters=256,
            depth_multiplier=depth_multiplier,
            strides=(1, 1),
            training=self.training
        )

        # 28 * 28 * 256 -> 14 * 14 * 512
        conv_7 = self.depth_wise_conv_block(
            inputs=conv_6,
            filters=256,
            point_wise_conv_filters=512,
            depth_multiplier=depth_multiplier,
            strides=(2, 2),
            training=self.training
        )

        # 14 * 14 * 512 -> 14 * 14 * 512
        conv_8 = self.depth_wise_conv_block(
            inputs=conv_7,
            filters=512,
            point_wise_conv_filters=512,
            depth_multiplier=depth_multiplier,
            strides=(1, 1),
            training=self.training
        )
        conv_8 = self.depth_wise_conv_block(
            inputs=conv_8,
            filters=512,
            point_wise_conv_filters=512,
            depth_multiplier=depth_multiplier,
            strides=(1, 1),
            training=self.training
        )
        conv_8 = self.depth_wise_conv_block(
            inputs=conv_8,
            filters=512,
            point_wise_conv_filters=512,
            depth_multiplier=depth_multiplier,
            strides=(1, 1),
            training=self.training
        )
        conv_8 = self.depth_wise_conv_block(
            inputs=conv_8,
            filters=512,
            point_wise_conv_filters=512,
            depth_multiplier=depth_multiplier,
            strides=(1, 1),
            training=self.training
        )
        conv_8 = self.depth_wise_conv_block(
            inputs=conv_8,
            filters=512,
            point_wise_conv_filters=512,
            depth_multiplier=depth_multiplier,
            strides=(1, 1),
            training=self.training
        )

        # 14 * 14 * 512 -> 7 * 7 * 1024
        conv_9 = self.depth_wise_conv_block(
            inputs=conv_8,
            filters=512,
            point_wise_conv_filters=1024,
            depth_multiplier=depth_multiplier,
            strides=(2, 2),
            training=self.training
        )

        # 7 * 7 * 1024 -> 1 * 1 * 1024
        outputs = tf.layers.average_pooling2d(
            inputs=conv_9,
            pool_size=(7, 7),
            strides=(1, 1),
            padding="VALID"
        )
        outputs = tf.layers.dense(
            inputs=outputs,
            units=output_shape,
            activation=tf.nn.softmax
        )
        outputs = tf.layers.dropout(outputs, 0.5)
        self.outputs = tf.squeeze(outputs, axis=[1, 2])

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.outputs, labels=self.y))
        self.train_op = tf.train.AdamOptimizer(1e-3).minimize(self.loss)
        self.correct_infer = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.outputs, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_infer, tf.float32))

    def train(self, sess, x_train, y_train, x_valid, y_valid, batch_size, epoch):
        for i in range(epoch):
            start_time = time.time()
            for batch_start in range(0, len(x_train), batch_size):
                batch_end = batch_start + batch_size
                batch_x = x_train[batch_start:batch_end]
                batch_y = y_train[batch_start:batch_end]

                sess.run(self.train_op, feed_dict={self.x: batch_x, self.y: batch_y, self.training: True})

            loss_value = sess.run(
                self.loss,
                feed_dict={
                    self.x: batch_x,
                    self.y: batch_y,
                    self.training: False
                })
            accuracy_value = sess.run(
                self.accuracy,
                feed_dict={
                    self.x: x_valid,
                    self.y: y_valid,
                    self.training: False
                })
            print('-------------%d次迭代---------------' % i)
            print('总训练时间：%.2f' % (time.time() - start_time))
            print('当前损失值：%.2f' % loss_value)
            print('当前准确率：%.2f' % accuracy_value)
        print('-------------训练结束---------------')

    def inference(self, sess, inputs):
        return sess.run(tf.argmax(self.outputs, axis=1), feed_dict={self.x: inputs})

    @staticmethod
    def conv_block(inputs, filters, kernel_size, strides, padding, training):
        conv = tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation=None
        )
        bn = tf.layers.batch_normalization(inputs=conv, training=training)
        res = tf.nn.relu6(bn)

        return res

    @staticmethod
    def depth_wise_conv_block(inputs, filters, point_wise_conv_filters, depth_multiplier, strides, training):
        conv = tf.layers.separable_conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=(3, 3),
            depth_multiplier=depth_multiplier,
            strides=strides,
            padding="SAME",
            use_bias=False,
        )
        bn = tf.layers.batch_normalization(inputs=conv, training=training)
        relu = tf.nn.relu6(bn)

        conv = tf.layers.conv2d(
            inputs=relu,
            filters=point_wise_conv_filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='SAME',
            use_bias=False
        )
        bn = tf.layers.batch_normalization(inputs=conv, training=training)
        return tf.nn.relu(bn)
