import tensorflow as tf
from keras.datasets import cifar100
from network import resnet_50
from keras.utils import to_categorical
import time

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../utils')))
from utils import image_resize

output_shape = 100
num_image = 1000
batch_size = 10
epoch = 50


def main():
    (images, labels), _ = cifar100.load_data()
    x_train = image_resize(images, (224, 224), num_image=num_image, normalize=False)
    x_valid = image_resize(images[num_image:], (224, 224), num_image=200, normalize=False)
    y_train = to_categorical(labels[:num_image], num_classes=output_shape)
    y_valid = to_categorical(labels[num_image:num_image + 200], num_classes=output_shape)

    x = tf.placeholder(tf.float32, [None, 224, 224, 3])
    y = tf.placeholder(tf.int32, [None, output_shape])
    is_training = tf.placeholder(tf.bool)

    outputs = resnet_50(inputs=x, output_shape=output_shape, is_training=is_training)

    # Loss function and optimizer
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

    # Accuracy
    correct_prediction = tf.equal(tf.argmax(outputs, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        start_time = time.time()
        for i in range(epoch):
            for batch_start in range(0, len(x_train), batch_size):
                batch_end = batch_start + batch_size
                batch_x = x_train[batch_start:batch_end]
                batch_y = y_train[batch_start:batch_end]

                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, is_training: True})

            loss_value = sess.run(loss, feed_dict={x: batch_x, y: batch_y, is_training: False})
            accuracy_value = sess.run(accuracy, feed_dict={x: x_valid, y: y_valid, is_training: False})
            print('-------------%d次迭代---------------' % i)
            print('总训练时间：%.2f' % (time.time() - start_time))
            print('当前损失值：%.2f' % loss_value)
            print('当前准确率：%.2f' % accuracy_value)
        print('-------------训练结束---------------')

        print('训练完成，模型正在保存……')
        saver.save(sess, './model/tf_model/resnet50.ckpt')
        print('模型保存成功')


if __name__ == '__main__':
    main()
