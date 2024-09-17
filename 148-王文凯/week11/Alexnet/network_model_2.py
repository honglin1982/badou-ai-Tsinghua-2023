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

        # 权重字典
        weight_dict = {
            'conv_1': tf.Variable(tf.truncated_normal([11, 11, 3, 96], stddev=1e-3)),
            'conv_2': tf.Variable(tf.truncated_normal([5, 5, 96, 256], stddev=1e-2)),
            'conv_3': tf.Variable(tf.truncated_normal([3, 3, 256, 384], stddev=1e-2)),
            'conv_4': tf.Variable(tf.truncated_normal([3, 3, 384, 384], stddev=1e-2)),
            'conv_5': tf.Variable(tf.truncated_normal([3, 3, 384, 256], stddev=1e-2)),
            'fc_1': tf.Variable(tf.truncated_normal([9216, 4096], stddev=1e-1)),
            'fc_2': tf.Variable(tf.truncated_normal([4096, 2048], stddev=1e-1)),
            'fc_3': tf.Variable(tf.truncated_normal([2048, output_shape], stddev=1e-1)),
        }

        # 偏移量字典
        bias_dict = {
            'conv_1': tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[96])),
            'conv_2': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[256])),
            'conv_3': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[384])),
            'conv_4': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[384])),
            'conv_5': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[256])),
            'fc_1': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[4096])),
            'fc_2': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[2048])),
            'fc_3': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[output_shape])),
        }

        # conv_1
        conv_1 = tf.nn.conv2d(self.x, weight_dict['conv_1'], strides=[1, 4, 4, 1], padding='VALID')
        conv_1 = tf.nn.bias_add(conv_1, bias_dict['conv_1'])
        relu_1 = tf.nn.relu(conv_1)
        pool_1 = tf.nn.max_pool(relu_1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

        # conv_2
        conv_2 = tf.nn.conv2d(pool_1, weight_dict['conv_2'], strides=[1, 1, 1, 1], padding='SAME')
        conv_2 = tf.nn.bias_add(conv_2, bias_dict['conv_2'])
        relu_2 = tf.nn.relu(conv_2)
        pool_2 = tf.nn.max_pool(relu_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

        # conv_3
        conv_3 = tf.nn.conv2d(pool_2, weight_dict['conv_3'], strides=[1, 1, 1, 1], padding='SAME')
        conv_3 = tf.nn.bias_add(conv_3, bias_dict['conv_3'])
        relu_3 = tf.nn.relu(conv_3)

        # conv_4
        conv_4 = tf.nn.conv2d(relu_3, weight_dict['conv_4'], strides=[1, 1, 1, 1], padding='SAME')
        conv_4 = tf.nn.bias_add(conv_4, bias_dict['conv_4'])
        relu_4 = tf.nn.relu(conv_4)

        # conv_5
        conv_5 = tf.nn.conv2d(relu_4, weight_dict['conv_5'], strides=[1, 1, 1, 1], padding='SAME')
        conv_5 = tf.nn.bias_add(conv_5, bias_dict['conv_5'])
        relu_5 = tf.nn.relu(conv_5)
        pool_4 = tf.nn.max_pool(relu_5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

        # 矩阵扁平化
        reshaped = tf.reshape(pool_4, [-1, 9216])

        # fc_1
        fc_1 = tf.add(tf.matmul(reshaped, weight_dict['fc_1']), bias_dict['fc_1'])
        fc_1_relu = tf.nn.relu(fc_1)
        fc_1_dropout = tf.nn.dropout(fc_1_relu, 0.5)

        # fc_2
        fc_2 = tf.add(tf.matmul(fc_1_dropout, weight_dict['fc_2']), bias_dict['fc_2'])
        fc_2_relu = tf.nn.relu(fc_2)
        fc_2_dropout = tf.nn.dropout(fc_2_relu, 0.5)

        # output_res
        self.output_res = tf.add(tf.matmul(fc_2_dropout, weight_dict['fc_3']), bias_dict['fc_3'])

        # train set
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output_res, labels=self.y_))
        self.train_op = tf.train.AdamOptimizer(1e-3).minimize(self.loss)
        self.correct_infer = tf.equal(tf.argmax(self.y_, 1), tf.argmax(self.output_res, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_infer, tf.float32))

    def train(self, sess, data_train, labels_train, data_verify, labels_verify, batch_size, epoch, patience):
        len_data_train, len_data_verify = len(data_train), len(data_verify)
        # dataset_train
        dataset_train = tf.data.Dataset.from_tensor_slices((data_train, labels_train))
        dataset_train = dataset_train.shuffle(buffer_size=len_data_train)
        dataset_train = dataset_train.batch(batch_size)
        iterator_train = dataset_train.make_initializable_iterator()
        # 初始化迭代器
        sess.run(iterator_train.initializer)
        next_batch_train = iterator_train.get_next()

        # dataset_verify
        dataset_verify = tf.data.Dataset.from_tensor_slices((data_verify, labels_verify))
        dataset_verify = dataset_verify.shuffle(buffer_size=len_data_train)
        dataset_verify = dataset_verify.batch(batch_size)
        iterator_verify = dataset_verify.make_initializable_iterator()
        # 初始化迭代器
        sess.run(iterator_verify.initializer)
        next_batch_verify = iterator_verify.get_next()

        start_time = time.time()
        cnt_train, cnt_verify, max_acc = 0, 0, 0

        # early_stopping
        # 用于保存最佳的验证集损失值
        best_loss = float('inf')
        # 计数器，用于跟踪连续验证集损失没有下降的次数
        counter = 0

        for step in range(1, epoch + 1):
            cnt_train += batch_size
            cnt_verify += batch_size
            if cnt_train >= len_data_train:
                sess.run(iterator_train.initializer)
                cnt_train = 0
            if cnt_verify >= len_data_verify:
                sess.run(iterator_verify.initializer)
                cnt_verify = 0

            images_train_batch, labels_train_batch = sess.run(next_batch_train)
            labels_train_batch = sess.run(tf.one_hot(labels_train_batch, depth=self.output_shape))
            images_verify_batch, labels_verify_batch = sess.run(next_batch_verify)
            labels_verify_batch = sess.run(tf.one_hot(labels_verify_batch, depth=self.output_shape))

            train_feed_dict = {self.x: images_train_batch, self.y_: labels_train_batch}
            verify_feed_dict = {self.x: images_verify_batch, self.y_: labels_verify_batch}

            sess.run(self.train_op, feed_dict=train_feed_dict)
            loss_value = sess.run(self.loss, feed_dict=train_feed_dict)
            accuracy_value = sess.run(self.accuracy, feed_dict=verify_feed_dict)

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

