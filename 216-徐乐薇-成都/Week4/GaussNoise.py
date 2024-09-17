import numpy as np
import random
import cv2
from numpy import shape
import copy

#高斯噪声
def GaussianNoise(src, means, sigma, percent):
    NoiseImg = copy.copy(src)  #使用浅拷贝，赋值到新变量后与原图像无关。如果直接赋值经过该函数后会改变原图像src
    #NoiseImg = src
    NoiseNum = int(percent*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        randX = random.randint(0, src.shape[0] - 1) #随机生成的行
        randY = random.randint(0, src.shape[1] - 1) #随机生成的列
        NoiseImg[randX, randY] = src[randX, randY] + random.gauss(means, sigma) #在原有像素灰度值上加上随机生成的高斯噪声

        #范围约束，小于0设为0，大于255设为255
        if NoiseImg[randX, randY] < 0:
            NoiseImg[randX, randY] = 0
        elif NoiseImg[randX, randY] > 255:
            NoiseImg[randX, randY] = 255;
    return NoiseImg

img = cv2.imread('lenna.png', 0) #flag = 0时为8位深度，1通道(灰)，如果不加flag默认flag = 1,8位深度，3通道(彩)
img1 = GaussianNoise(img, 2.1, 7, 0.9)
cv2.imshow('source', img)
cv2.imshow('GaussianNoise', img1)
cv2.waitKey(0)

# img = cv2.imread('lenna.png', 0) #flag = 0时为8位深度，1通道(灰)，如果不加flag默认flag = 1,8位深度，3通道(彩)
# img1 = GaussianNoise(img, 2.1, 7, 0.9)
# img = cv2.imread('lenna.png', 1)
# img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('source', img2)
# cv2.imshow('GaussianNoise', img1)
# cv2.waitKey(0)