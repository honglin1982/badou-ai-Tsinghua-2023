from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image


# 1. 实现RGB2GRAY（手工实现）
image = cv2.imread("lenna.png")
h, w = image.shape[:2]  # 获取图片高、宽
image_gray = np.zeros([h, w], image.dtype)  # 创建单通道图片，高和宽与原图相同
for i in range(h):
    for j in range(w):
        m = image[i, j]  # 高和宽的BGR坐标
        image_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)  # BGR->灰度化计算公式，m[0]为B，m[1]为G，m[2]为R，cv2是BGR读取顺序，赋给新图像

cv2.imshow("image gray", image_gray)
print("图像灰度: %s" % image_gray)
print(image_gray)

# 原图
plt.subplot(131)
image = plt.imread("lenna.png")
plt.imshow(image)
print("lenna")
print(image)

# 2. 实现RGB2GRAY（调接口）
image_gray = rgb2gray(image)
plt.subplot(132)
plt.imshow(image_gray, cmap='gray')
print("---灰度图----")
print(image_gray)

# 3. 二值化
image_binary = np.where(image_gray >= 0.5, 1, 0)
print("-----二值化图像------")
print(image_binary)
print(image_binary.shape)

plt.subplot(133)
plt.imshow(image_binary, cmap='gray')
plt.show()
