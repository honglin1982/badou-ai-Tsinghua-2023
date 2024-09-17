import numpy as np
import random
import cv2
from numpy import shape
import copy

#椒盐噪声
def PepperSaltNoise(src, percent):
    NoiseImg = copy.copy(src)  #使用浅拷贝，赋值到新变量后与原图像无关。如果直接赋值经过该函数后会改变原图像src
    #NoiseImg = src
    NoiseNum = int(percent*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        randX = random.randint(0, src.shape[0] - 1) #随机生成的行
        randY = random.randint(0, src.shape[1] - 1) #随机生成的列

        #随机生成像素点一半为白一半为黑
        if random.random() <= 0.5:
            NoiseImg[randX, randY] = 0
        else:
            NoiseImg[randX, randY] = 255;
    return NoiseImg

img = cv2.imread('lenna.png', 0) #flag = 0时为8位深度，1通道(灰)，如果不加flag默认flag = 1,8位深度，3通道(彩)
img1 = PepperSaltNoise(img, 0.1)
cv2.imshow('source', img)
cv2.imshow('PepperSaltNoise', img1)
cv2.waitKey(0)
