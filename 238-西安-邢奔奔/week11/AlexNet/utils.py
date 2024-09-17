import matplotlib.image as mping
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.python.ops import array_ops


def load_image(path):
    image = mping.imread(path)
    short_edge = min(image.shape[:2])
    yy = int((image.shape[0] - short_edge) / 2)
    xx = int((image.shape[1] - short_edge) / 2)
    crop_img = image[yy:yy + short_edge, xx:xx + short_edge]
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
    with open("./index_word.txt", "r", encoding='utf-8') as f:
        # 将字符串划分后，取索引为1的字符串，并将末尾字符除去
        synset = [l.split(";")[1][:-1] for l in f.readlines()]

        print(synset[argmax])
        return synset[argmax]
