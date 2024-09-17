import matplotlib.image as mping
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops


def load_image(path):
    img = mping.imread(path)

    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:short_edge + yy, xx:short_edge + xx]
    return crop_img


def resize_img(image, size,
               method=tf.image.ResizeMethod.BILINEAR,
               align_corners=False):
    # align_corners=False用于控制调整图像大小时是否应该对齐角点。它用于控制调整图像大小时是否应该对齐角点。
    with tf.name_scope('Resize_image'):
        img = tf.expand_dims(image, 0)
        img = tf.image.resize_images(img, size, method, align_corners)
        # tf.stack([-1, size[0], size[1], 3]) 创建了一个形状为 [-1, size[0], size[1], 3] 的张量，其中 -1 表示自动推断的维度大小
        # 这样做是为了保持调整大小后的图像张量的形状和通道数不变，但自动调整批量大小（batch size）
        img = tf.reshape(img, tf.stack([-1, size[0], size[1], 3]))
    return img


def print_prob(prob, file_path):
    # l.strip() 是一个字符串的方法，用于去除字符串两端的空白字符（包括空格、制表符和换行符）
    synset = [l.strip() for l in open(file_path).readlines()]
    # [::-1]：用于反转数组的顺序
    pred = np.argsort(prob)[::-1]

    top1 = synset[pred[0]]
    print('top1:', top1, prob[pred[0]])
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print('top5:', top5)
