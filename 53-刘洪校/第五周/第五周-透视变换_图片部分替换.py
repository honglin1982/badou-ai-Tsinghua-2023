# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 14:50:53 2023

@author: lhx
"""

import cv2
import numpy as np


#图片内容替换
img = cv2.imread('small.jpg')
cv2.imshow("before", img)
bgImg = cv2.imread('big.png')

#源图坐标
src = np.float32([[0, 0], [540, 0], [0, 813], [540, 813]])
#目标图坐标 将图片变换为目标图上的坐标点
dst = np.float32([[177, 112], [376, 114], [180, 461], [378, 455]])

triangle = np.array([[177, 112], [376, 114], [378, 455], [180, 461]])#填充点的顺序左上，右上，右下，左下

# 生成透视变换矩阵；进行透视变换
m = cv2.getPerspectiveTransform(src, dst)
print("warpMatrix_change:")
print(m)
resultImg = cv2.warpPerspective(img, m, (940, 524))
cv2.imshow("after", resultImg)


cv2.fillConvexPoly(bgImg, triangle, (255, 255, 255))

cv2.imshow("beforeTarget", bgImg)
t=bgImg+resultImg
cv2.imshow("afterTarget", t)

cv2.waitKey(0)
cv2.destroyAllWindows()
