import tensorflow as tf
from network_model import AlexNet
import matplotlib.pyplot as plt
from data_processing import get_file
import os

batch_size = 128
train_epoch = 500


def main():
    model = AlexNet(input_shape=[227, 227, 3], output_shape=2)
    with tf.Session() as sess:
        save_model = './model/cat_and_dog'
        save_log = './log/cat_and_dog'
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
        save_plt += (os.sep + 'AlexNet_cat_and_dog.png')
        saver = tf.train.Saver()
        train_writer = tf.summary.FileWriter(save_log, sess.graph)
        data_train, labels_train = get_file('./data/image/train')

        sess.run(tf.global_variables_initializer())
        model.train(
            sess=sess,
            data_train=data_train,
            labels_train=labels_train,
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
