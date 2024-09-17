
#!/usr/bin/python
# encoding=gbk

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
img = cv.imread("r1.jpg", 1)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('image_gray', gray)

# 灰度图像直方图均衡化
dst = cv.equalizeHist(gray)

# 直方图
hist = cv.calcHist([dst],[0],None,[256],[0,256])

plt.figure()
plt.hist(dst.ravel(), 256)
plt.show()

cv.imshow("Histogram Equalization", np.hstack([gray, dst]))
cv.waitKey(0)
'''


# 彩色图像直方图均衡化
img = cv.imread("lenna.png", 1)
cv.imshow("src", img)

# 彩色图像均衡化,需要分解通道 对每一个通道均衡化
(b, g, r) = cv.split(img)
bH = cv.equalizeHist(b)
gH = cv.equalizeHist(g)
rH = cv.equalizeHist(r)
# 合并每一个通道
result = cv.merge((bH, gH, rH))
cv.imshow("dst_rgb", result)

cv.waitKey(0)