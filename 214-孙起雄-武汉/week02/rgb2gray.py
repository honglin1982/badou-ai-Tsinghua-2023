# 实现rgb2gray(手工+调接口)
# 1、调接口的方式实现
import cv2
import numpy as np
from PIL import Image
# 读取彩色图像
c_img = cv2.imread('7.jpg')
# 在窗口中显示图像，该窗口和图像的原始大小自适应
cv2.imshow('original image', c_img)
g_img = cv2.cvtColor(c_img, cv2.COLOR_RGB2GRAY)
# 将g_img 从二维矩阵转为image
gray = Image.fromarray(g_img)
# 保存图片，参数为保存的文件名
gray.save('gray.jpg')
cv2.imshow('Gray Image', g_img)
# 让窗口持久停留
cv2.waitKey(0)


# 2、手工实现灰度化
# 读取图片的高和宽
h, w = c_img.shape[:2]
# 创建同样大小的单通道图片
g_img = np.zeros([h, w], c_img.dtype)
for i in range(h):
    for j in range(w):
        # 取出BGR坐标
        m = c_img[i, j]
        # 将BGR坐标转化为gray坐标并赋给新图像
        g_img[i, j] = int(m[0]*0.11 + m[1]*0.59 +m[2]*0.3)
cv2.imshow("image show gray", g_img)
# 让窗口持久停留
cv2.waitKey(0)