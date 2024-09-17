import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf

def load_image(path):
    img = mpimg.imread(path)
    #裁剪图片为正方形
    shortEdge = min(img.shape[:2])
    ny = int((img.shape[0] - shortEdge) / 2)
    nx = int((img.shape[1] - shortEdge) / 2)
    cropImg = img[ny : ny + shortEdge, nx : nx + shortEdge]
    return cropImg

#调用双线性插值改变图像大小
def resize_image(image, size, method=tf.image.ResizeMethod.BILINEAR,
                 align_corners=False):
    with tf.name_scope('resize_image'):
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_images(image, size, method,
                                       align_corners)
        image = tf.reshape(image, tf.stack([-1, size[0], size[1], 3]))
        return image

def printProb(prob, filePath):

    synset = [l.strip() for l in open(filePath).readlines()]
    pred = np.argsort(prob)[::-1] #将概率从大到小排列的序号存入pred
    top1 = synset[pred[0]]
    print("Top1:", top1, prob[pred[0]])
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print("Top5:", top5)
    return top1
