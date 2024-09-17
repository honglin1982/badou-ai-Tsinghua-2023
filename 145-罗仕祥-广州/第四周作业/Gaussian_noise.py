#!/usr/bin/env python
# -*- coding: utf-8 -*-


import cv2
import random

def GaussianNoise(src, means, sigma, percetage, channel):
    NoiseImg = src
    NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(NoiseNum):
        # 每次取一个随机点
        # 把一张图片的像素用行和列表示的话，randX 代表随机生成的行，randY代表随机生成的列
        # random.randint生成随机整数
        # 高斯噪声图片边缘不处理，故-1
        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)
        # 此处在原有像素灰度值上加上随机数
        NoiseImg[randX, randY] = NoiseImg[randX, randY] + random.gauss(means, sigma)
        # 若灰度值小于0则强制为0，若灰度值大于255则强制为255
        if channel == 1:
            if NoiseImg[randX, randY] < 0:
                NoiseImg[randX, randY] = 0
            elif NoiseImg[randX, randY] > 255:
                NoiseImg[randX, randY] = 255
        elif channel == 3:
            for c in range(3):
                if NoiseImg[randX, randY, c] < 0:
                    NoiseImg[randX, randY, c] = 0
                elif NoiseImg[randX, randY, c] > 255:
                    NoiseImg[randX, randY, c] = 255

    return NoiseImg

channel = 3 # 1表示灰度图像，3表示彩色图像
if channel == 1:
    img = cv2.imread('lenna.png', 0)    # 读取灰度图像, 0表示灰度图像, 1表示彩色图像
elif channel == 3:
    img = cv2.imread('lenna.png', 1)    # 读取灰度图像, 0表示灰度图像, 1表示彩色图像
img1 = img.copy()
cv2.imshow('source', img)
img1 = GaussianNoise(img1, 2, 4, 0.6, channel)
cv2.imshow('lenna_GaussianNoise_0.6', img1)
img2 = img.copy()
img2 = GaussianNoise(img1, 2, 4, 0.3, channel)
cv2.imshow('lenna_GaussianNoise_0.3', img2)
cv2.waitKey(0)




# 比较两种噪声的效果，通过噪声的系数可以看出，椒盐噪声的系数越大，噪声越多，而高斯噪声的系数越大，噪声越少
# 通过噪声的均值和方差可以看出，椒盐噪声的均值和方差都为0，而高斯噪声的均值为2，方差为4
# 通过噪声的比例可以看出，椒盐噪声的比例越大，噪声越多，而高斯噪声的比例越大，噪声越少
# 通过噪声的图片可以看出，椒盐噪声的图片噪声点分布均匀，而高斯噪声的图片噪声点分布不均匀
# 通过噪声的图片可以看出，椒盐噪声的图片噪声点为黑点和白点，而高斯噪声的图片噪声点为灰度值
# 通过噪声的图片可以看出，椒盐噪声的图片噪声点为孤立点，而高斯噪声的图片噪声点为连续点





