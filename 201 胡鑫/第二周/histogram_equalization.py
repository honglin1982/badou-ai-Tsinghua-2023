import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../lenna.png') # cv2.imread读取图片的默认第二参数为1，即读取彩色图片
# 灰度化
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 原来的直方图
hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
# 灰度均衡化
dst_img = cv2.equalizeHist(img_gray)
# 均衡化后的直方图
dst_hist = cv2.calcHist([dst_img], [0], None, [256], [0, 256])

# # 使用plt.plot(hist)直接展示直方图，图像是连续的，不能按像素级展示数据
# plt.figure()
# plt.plot(hist)

# 展示原来的直方图
plt.figure(num='before')
plt.hist(img_gray.ravel(), 256) # 第一个参数只能为1维数组，使用ravel函数转换；第二个参数表示有256个像素级，表示需用256个部分展示直方图，每个部分就是网上资料上说的bin

# 展示均衡化后的直方图
plt.figure(num='after')
plt.hist(dst_img.ravel(), 256)
plt.show()

# 展示均衡化前后图像的差距
cv2.imshow('histogram equalization', np.hstack([img_gray, dst_img]))
cv2.waitKey()

# # 彩色图片均衡化
# img = cv2.imread('../lenna.png')
# # 分割通道
# b, g, r = cv2.split(img)
# # 针对每个通道分别进行灰度直方图均衡化
# bH = cv2.equalizeHist(b)
# gH = cv2.equalizeHist(g)
# rH = cv2.equalizeHist(r)
# # 均衡化后再合并
# img_hist = cv2.merge((bH, gH, rH))
# # 展示均衡化效果
# cv2.imshow('colorful equalization', np.hstack([img, img_hist]))
# cv2.waitKey()