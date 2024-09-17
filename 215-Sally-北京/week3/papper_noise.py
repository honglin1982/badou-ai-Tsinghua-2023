# -*- coding: utf-8 -*-

import cv2
import random

# 加入椒盐噪声
def papper(img, percentage):
    h, w = img.shape
    noiseImg = img.copy()
    noiseNum = int(h * w * percentage)

    for i in range(noiseNum):
        randX = random.randint(0, w - 1) # 随机取横坐标
        randY = random.randint(0, h - 1) # 随机取纵坐标
        noiseImg[randX, randY] = 0 if random.random() < 0.5 else 255 # 随机生成黑白点

    return noiseImg


if __name__ == '__main__':
    img = cv2.imread('./lenna.png', 0)
    dst = papper(img, 0.02)
    cv2.imshow('source', cv2.cvtColor(cv2.imread('./lenna.png'), cv2.COLOR_BGR2GRAY))
    cv2.imshow('lenna_PepperandSalt', dst)
    cv2.waitKey(0)
