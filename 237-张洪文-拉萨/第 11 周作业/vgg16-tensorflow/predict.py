from network import vgg16
import tensorflow as tf
import numpy as np
import utils


# 读取图片并修改
img = utils.load_image("./data/table.jpg")
inputs = tf.placeholder(tf.float32, shape=[None, None, 3])  # None表示任意数据
image_op = utils.resize_image(inputs, (224,224))

# 建立网络结构
prediction = vgg16.vgg_16(image_op)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # 初始化
    # 加载模型参数
    ckpt_filename = "./model/vgg_16.ckpt"
    saver = tf.train.Saver()  # 创建对象，用于恢复模型参数
    saver.restore(sess=sess, save_path=ckpt_filename)  # 加载

    # 对最后结果进行softmax预测，1000类
    pro = tf.nn.softmax(prediction)
    predict_data = sess.run(pro, feed_dict={inputs:img})  # 传递图片数据

    print("result: ")
    utils.print_prob(predict_data[0], "./data/synset.txt")


