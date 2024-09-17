# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 17:36:51 2023

@author: lhx
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

# =============================================================================
#  第一部分：灰度图直方图均衡化
# =============================================================================

img = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 灰度图直方图均衡化-使用cv2.equalizeHist(img)均衡化像素
dst = cv2.equalizeHist(gray)

# 直方图 
hist = cv2.calcHist([dst],[0],None,[256],[0,256])
print(hist)

print(img.ravel())
print(gray.ravel())
print(dst.ravel())

# 使用plt.hist绘制像素直方图
plt.figure(figsize=(15,3)) #不指定时默认大小，单位英寸

plt.subplot(131)
plt.hist(img.ravel(), 256)#256为直方图条形的个数
plt.subplot(132)
plt.hist(gray.ravel(), 256)
plt.subplot(133)
plt.hist(dst.ravel(), 256)
plt.show()
 
# 分别展示
#cv2.imshow("gray", gray)
#cv2.imshow("dst", dst)

# 横向一起展示
cv2.imshow("gray->dst", np.hstack((gray,dst)))
# 纵向一起展示
#cv2.imshow("gray->dst", np.vstack([gray,dst]))

#cv2.waitKey(0)
#cv2.destroyAllWindows()



# =============================================================================
#  第二部分：彩色图像直方图均衡化
# =============================================================================

img = cv2.imread("lenna.png", 1)
#cv2.imshow("src", img)

# 彩色图像均衡化,需要分解通道 对每一个通道均衡化
(b, g, r) = cv2.split(img)#通道拆分，cv2.split(img[, mv])，可指定通道B/G/R
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
# 合并每一个通道
dstRgb = cv2.merge((bH, gH, rH))
#cv2.imshow("dst_rgb", dstRgb)
cv2.imshow("RGB:gray->dst", np.hstack((img,dstRgb)))

cv2.waitKey(0)
cv2.destroyAllWindows()


# =============================================================================
# figure(num=None, figsize=None, dpi=None, facecolor=None, edgecolor=None, frameon=True)
# num:图像编号或名称，数字为编号 ，字符串为名称
# figsize:指定figure的宽和高，单位为英寸；
# dpi参数指定绘图对象的分辨率，即每英寸多少个像素，缺省值为80 1英寸等于2.5cm,A4纸是 21*30cm的纸张
# facecolor:背景颜色
# edgecolor:边框颜色
# frameon:是否显示边框
# =============================================================================
