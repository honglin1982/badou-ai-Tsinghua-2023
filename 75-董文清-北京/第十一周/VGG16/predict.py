from nets import vgg16
import tensorflow as tf
import numpy as np
import utils

#读取图片
img1 = utils.load_image("./test_data/table.jpg")
#resize 图片
inputs = tf.placeholder(tf.float32, [None, None, 3])
resizeImg = utils.resize_image(inputs, (224, 224))
#建立网络
prediction = vgg16.vgg_16(resizeImg)
#载入模型
sess = tf.Session()
ckptFileName = './model/vgg_16.ckpt'
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, ckptFileName)
#进行softmax预测
pro = tf.nn.softmax(prediction)
pre = sess.run(pro, feed_dict={inputs:img1})
#打印结果
print("result: ")
utils.printProb(pre[0], './synset.txt')