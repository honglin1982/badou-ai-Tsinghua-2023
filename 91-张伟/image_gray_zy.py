import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

img = cv2.imread("lenna.png")   #读取图片
h, w = img.shape[:2]  # 获取图片的high和wide
img_gray = np.zeros([h, w], img.dtype)  # 创建一张和当前图片大小一样的单通道图片
for i in range(h):
    for j in range(w):
        m = img[i, j]  # 取出当前high和wide中的BGR坐标
        img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)  # 浮点算法:将BGR坐标转化为gray坐标并赋值给新图像
# 原图
plt.subplot(221)
img = plt.imread("lenna.png")
plt.imshow(img)
img_gray = rgb2gray(img)
#灰度化
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
# 二值化
img_binary = np.where(img_gray >= 0.5, 1, 0)
plt.subplot(223)
plt.imshow(img_binary, cmap='gray')
plt.show()