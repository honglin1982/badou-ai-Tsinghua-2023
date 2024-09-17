# -*- coding: utf-8 -*-
"""

彩色图像转灰度图像

binary
"""
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
#手动灰度化
dst = cv2.imread("lenna.png")   #cv2 读入通道为bgr
h,w= dst.shape[:2]  #shape[H,W,C]提取前两位
dst_gray = np.zeros([h,w],dst.dtype)
for i in range(h):
    for j in range(w):
        m = dst[i,j]
        dst_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)  #将每个通道转化
print("gray image")
print(dst_gray)
plt.subplot(221)
plt.imshow(dst_gray, cmap='gray')

plt.subplot(222)
img = plt.imread("lenna.png")
plt.imshow(img)
print("input image")
print(img)

#手动二值化二值
"""
先将图片转化为灰度图
"""
dst_gray = rgb2gray(img)     #灰度转化函数调用
rows, cols = dst_gray.shape
for i in range(rows):
    for j in range(cols):
        if (dst_gray[i,j] <= 0.5):  #二值化阈值判断
            dst_gray[i,j] = 0;
        else:
            dst_gray[i,j] = 1;

plt.subplot(223)
plt.imshow(dst_gray, cmap='gray')
print("binary image")
print(dst_gray)

"""
调用二值化接口
"""
dst = cv2.imread("lenna.png")
dst_g = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
ret, th1 = cv2.threshold(dst_g,127,255,cv2.THRESH_BINARY)
plt.subplot(224)
plt.imshow(th1, cmap='gray')
plt.show()



