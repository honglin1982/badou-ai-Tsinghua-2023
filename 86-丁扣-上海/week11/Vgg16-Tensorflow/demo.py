import tensorflow as tf
import numpy as np
import utils
import cv2
from nets import vgg16

tf.compat.v1.disable_eager_execution()


if __name__ == '__main__':
    # 对输入的图片进行resize，使其shape满足(-1,224,224,3)
    inputs = tf.compat.v1.placeholder(tf.float32, [None, None, 3])
    resize_img = utils.resize_image(inputs, (224, 224))
    print(resize_img.shape)

    # 建立网络结构
    prediction = vgg16.vgg_16(resize_img)

    # 载入模型
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    ckpt_filename = './model/vgg_16.ckpt'
    saver = tf.compat.v1.train.Saver()  # 模型存储，可以将新训练存储到原先的文件中
    saver.restore(sess, ckpt_filename)  # 加载

    # 读取图片
    # img1 = utils.load_image(r'./test_data/dog.jpg')
    img1 = utils.load_image(r'./test_data/table.jpg')
    print(img1.shape)

    # 最后结果进行softmax预测
    pro = tf.nn.softmax(prediction)
    res = sess.run(pro, feed_dict={inputs: img1})
    print(len(res))  # 1
    print(utils.print_prob(res[0], r'./synset.txt'))








