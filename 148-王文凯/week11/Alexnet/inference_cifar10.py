import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import numpy as np
from network_model import AlexNet

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../utils')))
from utils import get_cifar_10_label, image_resize


def main():
    _, (data_test, labels_test) = cifar10.load_data()
    data_test = image_resize(data_test, (227, 227))
    labels_test = np.array([item[0] for item in labels_test])
    model = AlexNet(input_shape=[227, 227, 3], output_shape=10)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        save_model = tf.train.latest_checkpoint('./model/cifar10')
        saver.restore(sess, save_model)
        inference_res = model.inference(sess, data_test[:100])
        score = []
        for i in range(100):
            score.append(1 if inference_res[i] == labels_test[i] else 0)
            print('真实结果：', get_cifar_10_label(labels_test[i])['CN'])
            print('模型推断结果：', get_cifar_10_label(inference_res[i])['CN'])

        print('准确率：%.2f' % (float(sum(score)) / 100))


if __name__ == '__main__':
    main()
