import tensorflow as tf
from network_model import AlexNet
from data_processing import get_file


def main():
    images_list, labels_list = get_file('./data/image/test')
    model = AlexNet(input_shape=[227, 227, 3], output_shape=2)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        save_model = tf.train.latest_checkpoint('./model/cat_and_dog')
        saver.restore(sess, save_model)
        inference_res = model.inference(sess, images_list)
        score = []
        for i in range(len(images_list)):
            score.append(1 if inference_res[i] == labels_list[i] else 0)
            print('真实结果：', 'cat' if labels_list[i] == 0 else 'dog')
            print('模型推断结果：', 'cat' if inference_res[i] == 0 else 'dog')

        print('准确率：%.2f' % (float(sum(score)) / len(images_list)))


if __name__ == '__main__':
    main()
