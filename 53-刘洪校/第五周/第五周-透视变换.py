# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 14:50:53 2023

@author: lhx
"""

import cv2
import numpy as np

img1 = cv2.imread('photo1.jpg')

outImg1 = img1.copy()

'''
注意这里src和dst的输入并不是图像，而是图像对应的顶点坐标。
'''
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
print(img1.shape)
# 生成透视变换矩阵；进行透视变换
m1 = cv2.getPerspectiveTransform(src, dst)
print("warpMatrix:")
print(m1)
result1 = cv2.warpPerspective(outImg1, m1, (337, 488))
cv2.imshow("src", img1)
cv2.imshow("result", result1)

#图片内容替换
img = cv2.imread('small.jpg')
cv2.imshow("before", img)
bgImg = cv2.imread('big.png')

#源图坐标
src = np.float32([[0, 0], [540, 0], [0, 813], [540, 813]])
#目标图坐标
dst = np.float32([[177, 112], [376, 114], [180, 461], [378, 455]])

# 生成透视变换矩阵；进行透视变换
m = cv2.getPerspectiveTransform(src, dst)
print("warpMatrix_change:")
print(m)
resultImg = cv2.warpPerspective(img, m, (200, 350))
cv2.imshow("after", resultImg)

cv2.imshow("beforeTarget", bgImg)
t=bgImg+resultImg
cv2.imshow("afterTarget", t)

cv2.waitKey(0)
cv2.destroyAllWindows()
