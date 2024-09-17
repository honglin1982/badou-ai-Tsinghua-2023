from nets import VGG16
import tensorflow as tf
import utils


img1 = utils.load_image("./test_data/dog.jpg")
inputs = tf.placeholder(tf.float32, [None,None,3])
resize_img = utils.resize_img(inputs, (224,224))

prediction = VGG16.vgg_16(resize_img)

with tf.Session() as sess:
    ckpt_filename = './model/vgg_16.ckpt'
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_filename)
    pro = tf.nn.softmax(prediction)
    pre = sess.run(pro, feed_dict={inputs:img1})
    print("result：")
    utils.print_prob(pre[0], './synset.txt')
