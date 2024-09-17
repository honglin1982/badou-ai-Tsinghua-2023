import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from network import vgg_19
from data_processing import get_data
import time

num_img = 1000
batch_size = 10
epochs = 50
output_shape = 100
lr = 1e-4


def main():
    # 数据预处理
    x_train, y_train = get_data(tag='train', img_shape=(224, 224), size=num_img, label_process=False)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

    encoder = OneHotEncoder()
    y_train = encoder.fit_transform(y_train).toarray()
    y_valid = encoder.transform(y_valid).toarray()

    x = tf.placeholder(tf.float32, [None, 224, 224, 3])
    y = tf.placeholder(tf.float32, [None, output_shape])
    is_training = tf.placeholder(tf.bool)

    output_layer = vgg_19(x, output_shape=output_shape)

    # Loss function and optimizer
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    # Accuracy
    correct_prediction = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        start_time = time.time()
        for epoch in range(epochs):
            for batch_start in range(0, len(x_train), batch_size):
                batch_end = batch_start + batch_size
                batch_x = x_train[batch_start:batch_end]
                batch_y = y_train[batch_start:batch_end]

                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, is_training: True})

            loss_value = sess.run(loss, feed_dict={x: batch_x, y: batch_y, is_training: False})
            accuracy_value = sess.run(accuracy, feed_dict={x: x_valid, y: y_valid, is_training: False})
            print('-------------%d次迭代---------------' % epoch)
            print('总训练时间：%.2f' % (time.time() - start_time))
            print('当前损失值：%.2f' % loss_value)
            print('当前准确率：%.2f' % accuracy_value)
            if epoch < epochs:
                print('-------------训练继续---------------')
        print('-------------训练结束---------------')
        print('训练完成，模型正在保存……')
        saver.save(sess, './model/VGG_16.ckpt')
        print('模型保存成功')


if __name__ == '__main__':
    main()
