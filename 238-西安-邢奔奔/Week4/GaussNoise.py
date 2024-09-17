#!/usr/bin/python
# -*- conding:utf-8 -*-

import numpy as np
import cv2
from numpy import shape
import random

# 随机生成符合高斯分布的随机数，mean和sigma是两个参数
def GaussNoise(src, mean, sigma, percentage):
    Noiseimage = src
    Noisenum = int(src.shape[0] * src.shape[1] * percentage)
    #此处根据percentage来计算需要加噪的像素点个数
    for i in range(Noisenum):
        random_X = random.sample(range(0, src.shape[0] - 1), 1)
        random_Y = random.sample(range(0, src.shape[1] - 1), 1)
        # 此处产生随机数，已去重
        Noiseimage[random_X, random_Y] = Noiseimage[random_X, random_Y] + random.gauss(mean, sigma)
        # 此处给选中的像素点加噪
        if Noiseimage[random_X, random_Y] < 0:
            Noiseimage[random_X, random_Y] = 0
        elif Noiseimage[random_X, random_Y] > 255:
            Noiseimage[random_X, random_Y] = 255
            # 此处判断灰度值是否越界
    return Noiseimage


if __name__ == '__main__':
    img = cv2.imread('/Users/aragaki/artificial/image/lenna.png', 0)
    dstImage = GaussNoise(img, 2, 4, 0.8)
    img = cv2.imread('/Users/aragaki/artificial/image/lenna.png',0)
    # contrImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("ContrastImage", contrImage)
    cv2.imshow("ContrastImage",img)
    # img经过函数处理后已经改变了
    cv2.imshow("DstImage", dstImage)
    cv2.waitKey(0)
