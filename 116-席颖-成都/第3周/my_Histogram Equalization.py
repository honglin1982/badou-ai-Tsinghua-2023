#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt

def equalization_Hist(img):
    width,height,chans = img.shape
    result_img = np.zeros((width,height,chans),np.uint8)

    arr = np.zeros((chans, 256))
    for i in range(chans):
        for x in range(width):
            for y in range(height):
                arr[i][img[x, y, i]] += 1

        sum = 0
        for j in range(256):
            sum += arr[i][j]
            arr[i][j] = int(sum*256/(width*height) - 1)

        for x in range(width):
            for y in range(height):
                result_img[x, y, i] = arr[i][img[x, y, i]]

    return result_img

def color_equalization(img):
    chans = cv2.split(img)
    colors = ("r", "g", "b")
    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])


img = cv2.imread("lenna.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
result_img = equalization_Hist(img_rgb)

plt.figure()
plt.subplot(221)
plt.imshow(img_rgb)
plt.title("img_rgb figure")

plt.subplot(222)
plt.imshow(result_img)
plt.title("img_equalization figure")

plt.subplot(223)
color_equalization(img_rgb)
plt.title("img_rgb histogram figure")

plt.subplot(224)
color_equalization(result_img)
plt.title("img_equalization histogram figure")

plt.show()

"""
cv2的方式实现
'''
equalizeHist—直方图均衡化
函数原型： equalizeHist(src, dst=None)
src：图像矩阵(单通道图像)
dst：默认即可

'''

# 获取灰度图像
img = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow("image_gray", gray)

# 灰度图像直方图均衡化
dst = cv2.equalizeHist(gray)

# 直方图
hist = cv2.calcHist([dst],[0],None,[256],[0,256])

plt.figure()
plt.hist(dst.ravel(), 256)
plt.show()

cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
cv2.waitKey(0)


'''
# 彩色图像直方图均衡化
img = cv2.imread("lenna.png", 1)
cv2.imshow("src", img)

# 彩色图像均衡化,需要分解通道 对每一个通道均衡化
(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
# 合并每一个通道
result = cv2.merge((bH, gH, rH))
cv2.imshow("dst_rgb", result)

cv2.waitKey(0)
"""