import tensorflow as tf
from keras.datasets import cifar100
from network import Mobilenet
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
    x_valid = image_resize(images[num_image:], (224, 224), num_image=100, normalize=False)
    y_train = to_categorical(labels[:num_image], num_classes=output_shape)
    y_valid = to_categorical(labels[num_image:num_image + 100], num_classes=output_shape)

    model = Mobilenet(input_shape=[224, 224, 3], output_shape=output_shape, depth_multiplier=1)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        model.train(
            sess=sess,
            x_train=x_train,
            y_train=y_train,
            x_valid=x_valid,
            y_valid=y_valid,
            batch_size=batch_size,
            epoch=epoch
        )

        print('训练完成，模型正在保存……')
        saver.save(sess, './model/inception_v3.ckpt')
        print('模型保存成功')


if __name__ == '__main__':
    main()
