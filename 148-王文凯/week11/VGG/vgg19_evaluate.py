import tensorflow as tf
from data_processing import get_data
from network import vgg_19

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../utils')))
from utils import get_cifar_100_label


def main():
    images, labels = get_data(tag='evaluate', img_shape=(224, 224), size=500)
    model = vgg_19(input_shape=[224, 224, 3], output_shape=100)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        save_model = tf.train.latest_checkpoint('./model')
        saver.restore(sess, save_model)
        inference_res = model.inference(sess, images)
        score = []
        for i in range(100):
            score.append(1 if inference_res[i] == labels[i] else 0)
            print('真实结果：', get_cifar_100_label(labels[i])['CN'])
            print('模型推断结果：', get_cifar_100_label(inference_res[i])['CN'])

        print('准确率：%.2f' % (float(sum(score)) / 100))


if __name__ == '__main__':
    main()
