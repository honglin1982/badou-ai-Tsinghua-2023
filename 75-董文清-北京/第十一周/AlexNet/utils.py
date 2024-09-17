import matplotlib.image as mpimg
import numpy as np
import cv2
import tensorflow as tf


def loadImage(path):

    img = mpimg.imread(path)
    #将图片裁剪为中心的正方形
    shortEdge = min(img.shape[:2])
    ny = int((img.shape[0] - shortEdge) / 2)
    nx = int((img.shape[1] - shortEdge) / 2)
    cropImg = img[ny : ny + shortEdge, nx : nx + shortEdge]
    return cropImg

def resizeImage(img, size):
    with tf.name_scope('resize_image'):
        images = []
        for i in img :
            i = cv2.resize(i, size)
            images.append(i)
        images = np.array(images)
        return images

def printAnswer(argmax):
    with open('./data/model/index_word.txt', 'r', encoding='utf-8') as f:

            words = [l.split(';')[1][:-1] for l in f.readlines()]
            print(words[argmax])
            return words[argmax]