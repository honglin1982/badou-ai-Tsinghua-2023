# 实现椒盐噪声和高斯噪声

# 引入库
import cv2
import numpy as np

import random

def generate_unique_random_list(x, n):
    if n > x + 1:
        print("Error: Number of unique values requested exceeds the range.")
        return None

    numbers = list(range(x + 1))
    random.shuffle(numbers)

    return numbers[:n]

# 添加椒盐噪声设定参数 图片，椒噪声灰度，盐噪声灰度（默认0，255），噪声比例
def add_salt_and_pepper_noise(newimg,ratio):
    # 获取图像的高度、宽度和通道数
    height, width = newimg.shape

    # 获取一个不重复的随机数组
    random_list = generate_unique_random_list(height*width,int(height*width*ratio))

    for i in random_list:
        if (i % 2) ==1 :
            newimg[i//width,i%width]=0
        if (i % 2) ==0 :
            newimg[i//width,i%width]=255
    return newimg


img = cv2.imread('lenna.png', 0)
img_with_noise = add_salt_and_pepper_noise(img, 0.2)

cv2.imshow('source', img)
cv2.imshow('with_noise', img_with_noise)
cv2.waitKey(0)
cv2.destroyAllWindows()