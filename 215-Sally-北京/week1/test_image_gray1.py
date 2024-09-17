# -*- coding: utf-8 -*-

"""
彩色图像的灰度化、二值化
"""

from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# 1. 灰度化
img = cv2.imread('xiniu.png')

# 1.1 灰度化-手动
# 获取图片的height和width
h, w = img.shape[:2]
# print(h, w) # 649 1150huiduhua

# 创建一张和当前图片大小一样的单通道图片
img_gray = np.zeros([h, w], img.dtype)

for i in range(h):
    for j in range(w):
        # 取出当前h和w中的BGR坐标
        m = img[i, j]

        # 将BGR坐标转化为gray坐标并赋值给新图像
        img_gray[i, j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)

# print(img_gray)
# print("image show gray: %s"%img_gray)
cv2.imshow("image show gray", img_gray)


plt.subplot(221)
img = plt.imread('xiniu.png')
plt.imshow(img)
print("------demo image--------")
print(img)

# 1.2 灰度化-调用函数
img = cv2.imread('xiniu.png')
img_gray = rgb2gray(img)
# img_gray = cv2.cvtColor(img.cv2.COLOR_BGR2GRAY)
# img_gray = img
plt.subplot(222)
plt.imshow(img_gray, cmap="gray")
print("--------image gray---------")
print(img_gray)

# 2. 二值化
# 2.1 二值化-手动
rows, cols = img_gray.shape
for i in range(rows):
    for j in range(cols):
        img_gray[i, j] = 0 if img_gray[i, j] <= 0.5 else 1
# cv2.imshow("img binary show", img_gray)

# 2.2 二值化-调用函数
img_binary = np.where(img_gray >= 0.5, 1, 0)
print("--------image binary---------")
print(img_binary)
print(img_binary.shape)
cv2.imshow("img binary show", img_binary)

plt.subplot(223)
plt.imshow(img_binary, cmap='gray')
plt.show()
