#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import  numpy as np
from matplotlib import pyplot as plt
from timeit import default_timer as timer

img = cv2.imread('lenna.png', 1)
start = timer()

# 1.获取灰度图像
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# 2.灰度图像直方图均衡化
img_dst = cv2.equalizeHist(img_gray)

# 3.直方图
hist = cv2.calcHist([img_dst],[0],None,[256],[0,256])
plt.figure()
plt.hist(img_dst.ravel(), 256)
plt.show()

cv2.imshow("Histogram Equalization", np.hstack([img_gray, img_dst]))
cv2.imwrite('cv2_for_hist_equal.jpg',np.hstack([img_gray, img_dst]),[int(cv2.IMWRITE_JPEG_QUALITY),70])
print("with hist: ", timer() - start)
cv2.waitKey(0)