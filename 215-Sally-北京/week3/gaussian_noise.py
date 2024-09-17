# -*- coding: utf-8 -*-

import cv2
import random

# 加入高斯白噪声
# src - cv2.imread
# mean - 高斯分布参数
# sigma - 高斯分布参数
# percertage - 多少比例的点要做高斯分布
def gaussianNoise(src, mean, sigma, percertage):
    h, w = src.shape
    noiseImg = src
    noiseNum = int(w * h * percertage)

    for i in range(noiseNum):
        randX = random.randint(0, w - 1) # 随机取横坐标
        randY = random.randint(0, h - 1) # 随机取纵坐标
        noiseImg[randX, randY] = noiseImg[randX, randY] + random.gauss(mean, sigma) # PPT上的高斯噪声公式
        # 下面两行灰度纠错
        noiseImg[randX, randY] = max(noiseImg[randX, randY], 0)
        noiseImg[randX, randY] = min(noiseImg[randX, randY], 255)

        return noiseImg
    

if __name__ == '__main__':
    img = cv2.imread('./lenna.png')
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = gaussianNoise(imgGray, 2, 4, 0.9)

    cv2.imshow('source', imgGray)
    cv2.imshow('lenna_GaussianNoise', dst)
    cv2.waitKey(0)
