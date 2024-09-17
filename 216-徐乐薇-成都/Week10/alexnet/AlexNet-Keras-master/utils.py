import matplotlib.image as mpimg                #matplotlib.image模块中的imread函数，用于读取图片
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.python.ops import array_ops     #tensorflow.python.ops.array_ops模块中的pad函数，用于填充图片


def load_image(path):
    # 读取图片，rgb
    img = mpimg.imread(path)                    #读取图片
    # 将图片修剪成中心的正方形
    short_edge = min(img.shape[:2])             #取图片的最小边，因为要裁剪成正方形，所以取最小边，img.shape[:2]是取图片的高和宽,img.shape是取图片的高、宽和通道数
    yy = int((img.shape[0] - short_edge) / 2)   #计算裁剪的起始y坐标，因为要裁剪成正方形，所以起始y坐标是从最小边的一半开始。/2是为了取中间
    xx = int((img.shape[1] - short_edge) / 2)   #计算裁剪的起始x坐标，因为要裁剪成正方形，所以起始x坐标是从最小边的一半开始
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge] #裁剪图片
    return crop_img

def resize_image(image, size):
    with tf.name_scope('resize_image'):         #tf.name_scope()是为了更好的管理变量的命名空间，以便于在tensorboard中更好的可视化
        images = []
        for i in image:
            i = cv2.resize(i, size)             #将图片大小调整为size*size，cv2.resize()函数实现图片的缩放,参数size是一个tuple，格式为(width,height)
            images.append(i)                    #将调整后的图片添加到images中
        images = np.array(images)               #将images转化为数组
        return images

def print_answer(argmax):                       #定义一个函数，用于打印预测结果,argmax是预测结果
    with open("./data/model/index_word.txt","r",encoding='utf-8') as f: #with open(,r) 以读取的方式打开文件
        synset = [l.split(";")[1][:-1] for l in f.readlines()]          #f.readlines()是读取文件的每一行，l.split(";")是以分号分割每一行，[1]是取分号后面的内容，[:-1]是去掉最后的换行符
        
    print(synset[argmax])                                               #打印出预测结果
    return synset[argmax]