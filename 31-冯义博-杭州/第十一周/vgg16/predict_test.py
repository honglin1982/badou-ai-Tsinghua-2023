import vgg16_test
import utils
import tensorflow as tf


img = utils.load_image("./test_data/cat007.jpg")

inputs = tf.placeholder(tf.float32, [None, None, 3])
resized_img = utils.resize_image(inputs, (224, 224))

model = vgg16_test.vgg_16(resized_img)

with tf.Session() as session:
    ckpt_filename = './model/vgg_16.ckpt'
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(session, ckpt_filename)
    pro = tf.nn.softmax(model)
    pre = session.run(pro, feed_dict={inputs: img})


# 打印预测结果
print("result: ")
utils.print_prob(pre[0], './synset.txt')
