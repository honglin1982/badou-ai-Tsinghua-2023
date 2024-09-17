import tensorflow as tf
from nets import vgg16
import numpy as np
import utils

#载入图片
img = utils.load_image('./test_data/dog.jpg')
# 对输入的图片进行resize，使其shape满足(-1,224,224,3)
inputs = tf.placeholder(tf.float32,[None,None,3])
resized_image = utils.resize_image(inputs,(224,224))


#建立网络
prediction = vgg16.vgg_16(resized_image)


# 载入模型
sess = tf.Session()
ckpt_filename = './model/vgg_16.ckpt'
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, ckpt_filename)

# 最后结果进行softmax预测
pro = tf.nn.softmax(prediction)
pre = sess.run(pro,feed_dict={inputs:img})

#打印结果
print('result:')
utils.print_prob(pre[0],'./synset.txt')