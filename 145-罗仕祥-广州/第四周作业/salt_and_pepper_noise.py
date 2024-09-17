#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import random

def SaltAndPepper(src, percetage, channel):
    NoiseImg = src
    NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(NoiseNum):
        # 每次取一个随机点
        # 把一张图片的像素用行和列表示的话，randX 代表随机生成的行，randY代表随机生成的列
        # random.randint生成随机整数
        # 椒盐噪声图片边缘不处理，故-1
        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)
        # random.random生成随机浮点数，随意取到一个像素点有一半的可能是白点255，一半的可能是黑点0
        if channel == 1:
            if random.random() <= 0.5:
                NoiseImg[randX, randY] = 0
            else:
                NoiseImg[randX, randY] = 255
        elif channel == 3:
            for c in range(3):
                if random.random() <= 0.5:
                    NoiseImg[randX, randY, c] = 0
                else:
                    NoiseImg[randX, randY, c] = 255
    return NoiseImg

channel = 3 # 1表示灰度图像，3表示彩色图像
if channel == 1:
    img = cv2.imread('lenna.png', 0)
else:
    img = cv2.imread('lenna.png')
cv2.imshow('source', img)
img1 = img.copy()
img1 = SaltAndPepper(img1, 0.6, channel)
cv2.imshow('lenna_SaltAndPepper_0.6', img1)
img2 = img.copy()
img2 = SaltAndPepper(img2, 0.3, channel)
cv2.imshow('lenna_SaltAndPepper_0.3', img2)
cv2.waitKey(0)

