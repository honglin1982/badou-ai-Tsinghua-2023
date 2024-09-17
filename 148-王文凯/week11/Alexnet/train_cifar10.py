from tensorflow.keras.datasets import cifar10
import tensorflow as tf
from network_model import AlexNet
import numpy as np
import matplotlib.pyplot as plt
# from keras.utils import to_categorical

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../utils')))
from utils import get_cifar_10_label, image_resize

batch_size = 32
train_epoch = 500
num_image = 3000


def main():
    (data_train, labels_train), _ = cifar10.load_data()
    model = AlexNet(input_shape=[227, 227, 3], output_shape=10)
    data_train = image_resize(data_train, (227, 227), num_image)
    # labels_train = to_categorical(labels_train)
    labels_train = np.array([item[0] for item in labels_train])
    with tf.Session() as sess:
        save_model = './model/cifar10'
        save_log = './log/cifar10'
        save_plt = './plt'
        if not os.path.exists(save_model):
            print('模型保存目录{}不存在，正在创建……'.format(save_model))
            os.mkdir(save_model)
            print('创建成功')
        if not os.path.exists(save_log):
            print('日志保存目录{}不存在，正在创建……'.format(save_log))
            os.mkdir(save_log)
            print('创建成功')
        if not os.path.exists(save_plt):
            print('损失可视化保存目录{}不存在，正在创建……'.format(save_plt))
            os.mkdir(save_plt)
            print('创建成功')
        save_model += (os.sep + 'AlexNet.ckpt')
        save_plt += (os.sep + 'AlexNet.png')
        saver = tf.train.Saver()
        train_writer = tf.summary.FileWriter(save_log, sess.graph)

        sess.run(tf.global_variables_initializer())
        model.train(
            sess=sess,
            data_train=data_train,
            labels_train=labels_train[:num_image],
            batch_size=batch_size,
            epoch=train_epoch,
            patience=50
            )

        print('训练完成，模型正在保存……')
        saver.save(sess, save_model)
        print('模型保存成功')
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(model.losses)
        plt.xlabel('epoch')
        plt.ylabel('loss')

        plt.subplot(1, 2, 2)
        plt.plot(model.acc)
        plt.xlabel('epoch')
        plt.ylabel('acc')

        plt.tight_layout()
        plt.savefig(save_plt, dpi=200)
        plt.show()


if __name__ == "__main__":
    main()
