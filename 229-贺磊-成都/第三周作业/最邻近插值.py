# -*- coding: utf-8 -*-
# File  : 最邻近插值.py
# Author: HeLei
# Date  : 2023/4/28


import cv2
import numpy as np


def function(img):
    height, width, channels = img.shape  # lenna.png是512*512*3
    emptyImage = np.zeros((800, 800, channels), np.uint8)
    sh = 800 / height
    sw = 800 / width
    for i in range(800):
        for j in range(800):
            x = int(i / sh + 0.5)  # +0.5是为了向上取整,因为int()默认是向下取整。
            # 比如：3.4离3很近，使用int()后应该取3较好，但是3.7离4很近，应该取4，
            # 如果不加0.5，使用int()后向下取整就是3，所以加0.5后就是4。
            y = int(j / sw + 0.5)
            # x = int(i/sh)
            # y = int(j/sw)
            emptyImage[i, j] = img[x, y]  # 插入的是距离原图最近的点
    return emptyImage


# opencv封装好的函数
# cv2.resize(img,(800,800,c),near/bin)


img = cv2.imread("../data/lenna.png")
print(img.shape)
zoom = function(img)
print(zoom)
print(zoom.shape)
cv2.imshow("nearest interp", zoom)
cv2.imshow("image", img)
cv2.waitKey(0)
