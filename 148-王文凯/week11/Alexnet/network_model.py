import tensorflow as tf
import time


class AlexNet:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.x = tf.placeholder(tf.float32, shape=[None] + input_shape)
        self.y_ = tf.placeholder(tf.int32, shape=[None, output_shape])
        self.losses = []
        self.acc = []

        # conv_1
        conv_1 = tf.layers.conv2d(
            self.x,
            filters=96,
            kernel_size=[11, 11],
            strides=(4, 4),
            padding='VALID',
            activation=tf.nn.relu
        )
        # print('conv_1', conv_1.shape)
        pool_1 = tf.layers.max_pooling2d(
            conv_1,
            pool_size=[3, 3],
            strides=(2, 2),
            padding='VALID'
        )
        # print('pool_1', pool_1.shape)

        # conv_2
        conv_2 = tf.layers.conv2d(
            pool_1,
            filters=256,
            kernel_size=[5, 5],
            padding='SAME',
            activation=tf.nn.relu
        )
        pool_2 = tf.layers.max_pooling2d(
            conv_2,
            pool_size=[3, 3],
            strides=(2, 2),
            padding='VALID'
        )
        # print('pool_2', pool_2.shape)

        # conv_3
        conv_3 = tf.layers.conv2d(
            pool_2,
            filters=384,
            kernel_size=[3, 3],
            padding='SAME',
            activation=tf.nn.relu
        )
        # print('conv_3', conv_3.shape)

        # conv_4
        conv_4 = tf.layers.conv2d(
            conv_3,
            filters=384,
            kernel_size=[3, 3],
            padding='SAME',
            activation=tf.nn.relu
        )
        # print('conv_4', conv_4.shape)

        # conv_5
        conv_5 = tf.layers.conv2d(
            conv_4,
            filters=256,
            kernel_size=[3, 3],
            padding='SAME',
            activation=tf.nn.relu
        )

        pool_3 = tf.layers.max_pooling2d(
            conv_5,
            pool_size=[3, 3],
            strides=(2, 2),
            padding='VALID'
        )
        # print('pool_3', pool_3.shape)

        # 矩阵扁平化
        reshaped = tf.reshape(pool_3, [-1, 9216])

        # fc
        fc_1 = tf.layers.dense(
            inputs=reshaped,
            units=4096,
            activation=tf.nn.relu
        )
        fc_1 = tf.nn.dropout(fc_1, 0.5)
        # print('fc_1', fc_1.shape)

        fc_2 = tf.layers.dense(
            inputs=fc_1,
            units=4096,
            activation=tf.nn.relu
        )
        fc_2 = tf.nn.dropout(fc_2, 0.5)

        self.output_res = tf.layers.dense(
            fc_2,
            units=output_shape,
            activation=None
        )

        # train set
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output_res, labels=self.y_))
        self.train_op = tf.train.AdamOptimizer(1e-3).minimize(self.loss)
        self.correct_infer = tf.equal(tf.argmax(self.y_, 1), tf.argmax(self.output_res, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_infer, tf.float32))

    def train(self, sess, data_train, labels_train, batch_size, epoch, patience):
        len_data = len(data_train)
        # dataset
        dataset = tf.data.Dataset.from_tensor_slices((data_train, labels_train))
        dataset = dataset.shuffle(buffer_size=len_data)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_initializable_iterator()
        # 初始化迭代器
        sess.run(iterator.initializer)
        next_batch = iterator.get_next()

        start_time = time.time()
        cnt_train, max_acc = 0, 0

        # early_stopping
        # 用于保存最佳的验证集损失值
        best_loss = float('inf')
        # 计数器，用于跟踪连续验证集损失没有下降的次数
        counter = 0

        for step in range(1, epoch + 1):
            cnt_train += batch_size
            # cnt_verify += batch_size
            if cnt_train >= len_data:
                sess.run(iterator.initializer)
                cnt_train = 0

            images_batch, labels_batch = sess.run(next_batch)
            labels_batch = sess.run(tf.one_hot(labels_batch, depth=self.output_shape))

            feed_dict = {self.x: images_batch, self.y_: labels_batch}

            _, loss_value, accuracy_value = sess.run(
                [self.train_op, self.loss, self.accuracy],
                feed_dict=feed_dict
            )

            self.losses.append(loss_value)
            self.acc.append(accuracy_value)
            max_acc = max(max_acc, accuracy_value)

            print('-------------%d次迭代---------------' % step)
            if step % 10 == 0:
                print('-------------训练记录---------------')
                print('迭代次数：', step)
                print('总训练时间：%.2f' % (time.time() - start_time))
                print('当前损失值：%.2f' % loss_value)
                print('当前正确率：%.2f' % accuracy_value)
                print('当前最大正确率：%.2f' % max_acc)
                if step < epoch:
                    print('-------------训练继续---------------')

            # 检查是否满足 Early Stopping 条件
            if loss_value < best_loss:
                best_loss = loss_value
                # 重置计数器
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                print('Early Stopping at epoch', step)
                break

        print('-------------训练结束---------------')

    def inference(self, sess, input_data):
        return sess.run(tf.argmax(self.output_res, axis=1), feed_dict={self.x: input_data})

