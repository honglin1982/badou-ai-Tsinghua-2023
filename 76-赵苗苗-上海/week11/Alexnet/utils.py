import matplotlib.image as mpimg
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.python.ops import array_ops

#load_image()函数加载图像，对图像剪裁
def load_image(path):
    img=mpimg.imread(path)
    #将图片修剪成中心的正方形
    short_edge=min(img.shape[:2])  #计算图像最短边长度
    yy=int((img.shape[0]-short_edge)/2)  #(图像高度-最短边长度)
    xx=int((img.shape[1]-short_edge)/2)  #（图像宽度-最短边长度）
    crop_img=img[yy:yy+short_edge,xx:xx+short_edge]
    return crop_img

#resize_image()函数对图像做尺寸调整
def resize_image(image,size):
    with tf.name_scope('resize_image'):  #创建一个命名空间，将其中操作分组管理
        images=[]
        for i in image:
            i=cv2.resize(i,size)  #对当前遍历到的图像进行尺寸调整
            images.append(i)
        images=np.array(images)  #将images调整为多维数组
        return images

#print_answer()打开txt文件，根据argmax在文件中查找对应的类别标签
def print_answer(argmax):
    with open("./data/model/index_word.txt","r",encoding='utf-8') as f:  #打开文件并将其赋值给f
        synset=[l.split(',')[1][:-1]for l in f.readlines]  #读取文件每一行，并提取分割结果中的第二个元素

    print(synset[argmax])   #打印根据预测结果索引argmax在synset列表中得出的类别名称
    return synset[argmax]