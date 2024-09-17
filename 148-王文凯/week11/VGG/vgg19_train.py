import tensorflow as tf
from network import vgg_19
from data_processing import get_data
import time

output_shape = 100
img_size = 224
epoch = 500
batch_size = 32
train_size = 3000
lr = 1e-4
patience = 100

# 加载预训练模型
x = tf.placeholder(tf.float32, [None, img_size, img_size, 3])
y = tf.placeholder(tf.int32, [None, output_shape])
is_training = tf.placeholder(tf.bool)


def train():
    images, labels = get_data(tag='train', img_shape=(img_size, img_size), size=train_size)
    len_data = len(images)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(buffer_size=len_data)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    next_batch = iterator.get_next()
    best_loss = float('inf')
    counter = 0
    cnt = 0

    output_layer = vgg_19(inputs=x, output_shape=output_shape)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y))
    correct_infer = tf.equal(tf.argmax(y, 1), tf.argmax(output_layer, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_infer, tf.float32))
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        sess.run(iterator.initializer)
        start_time = time.time()
        saver = tf.train.Saver()
        for step in range(1, epoch + 1):
            counter += batch_size
            if counter >= len_data:
                sess.run(iterator.initializer)
                counter = 0
            batch_x, batch_y = sess.run(next_batch)
            batch_y = sess.run(tf.one_hot(batch_y, depth=output_shape))
            sess.run(train_op, feed_dict={x: batch_x, y: batch_y, is_training: True})
            accuracy_value, loss_value = sess.run(
                [accuracy, loss],
                feed_dict={x: batch_x, y: batch_y, is_training: False}
            )

            print('-------------%d次迭代---------------' % step)
            print('迭代次数：', step)
            print('总训练时间：%.2f' % (time.time() - start_time))
            print('当前损失值：%.2f' % loss_value)
            print('当前准确率：%.2f' % accuracy_value)
            # 检查是否满足 Early Stopping 条件
            if loss_value < best_loss:
                best_loss = loss_value
                # 重置计数器
                cnt = 0
            else:
                cnt += 1

            if cnt >= patience:
                print('Early Stopping at epoch', step)
                break
            if step < epoch:
                print('-------------训练继续---------------')
        print('-------------训练结束---------------')
        print('训练完成，模型正在保存……')
        saver.save(sess, './model/VGG_19.ckpt')
        print('模型保存成功')


if __name__ == '__main__':
    train()
