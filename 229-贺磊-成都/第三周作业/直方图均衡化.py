# -*- coding: utf-8 -*-
# File  : 直方图均衡化.py
# Author: HeLei
# Date  : 2023/5/1

import cv2
import numpy as np
from matplotlib import pyplot as plt


# 获取灰度图像
img = cv2.imread("../data/lenna.png", 1)  # cv2.imread(img,flag)flag=1表示彩图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度化
cv2.imshow("image_gray", gray)  # 灰度图

hist = cv2.calcHist([img], [0], None, [256], [0, 256])
plt.hist(img.ravel(), 256)
plt.show()  # 均衡化后的图

# 灰度图像直方图均衡化
dst = cv2.equalizeHist(gray)

# 直方图
hist1 = cv2.calcHist([dst], [0], None, [256], [0, 256])

plt.figure()
plt.hist(dst.ravel(), 256)
plt.show()

cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
cv2.waitKey(0)

'''
# 彩色图像直方图均衡化
img = cv2.imread("../data/lenna.png", 1)
# cv2.imshow("src", img)

# 彩色图像均衡化,需要分解通道 对每一个通道均衡化
(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
# 合并每一个通道
result = cv2.merge((bH, gH, rH))
imgStack = np.hstack((img, result))  # 相同大小图像水平拼接
cv2.imshow("dst_rgb", imgStack)

cv2.waitKey(0)
'''
