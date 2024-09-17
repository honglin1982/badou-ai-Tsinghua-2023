import tensorflow as tf
import numpy as np
import cv2
import matplotlib.image as mpimg
from tensorflow.python.ops import array_ops


def load_image(path):
    # 读取rgb图片
    img = mpimg.imread(path)
    # 将图片修剪成中心的正方形
    short_edge = min(img.shape[:2])
    h = int((img.shape[0] - short_edge) / 2)
    w = int((img.shape[1] - short_edge) / 2)
    crop_img = img[h:h+short_edge, w:w+short_edge]
    return crop_img


def resize_image(image, size):
    with tf.name_scope('resize_image'):
        images = []
        for i in image:
            i = cv2.resize(i, size)
            images.append(i)
        images = np.array(images)
        return images


def print_answer(argmax):
    with open('./data/model/index_word.txt', 'r', encoding='utf-8') as f:
        synset = [i.split(';')[1][:-1] for i in f.readlines()]
    print(synset[argmax])
    return synset[argmax]

