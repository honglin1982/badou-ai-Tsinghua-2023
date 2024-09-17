# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 10:50:20 2023

@author: lhx
"""
import cv2
import numpy as np

'''
cv2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient ]]])   
必要参数：
第一个参数是需要处理的原图像，该图像必须为单通道的灰度图；
第二个参数是滞后阈值1；
第三个参数是滞后阈值2。
'''


img = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("canny", cv2.Canny(gray, 22, 66))

img1 = cv2.imread("lenna.png", 0)
cv2.imshow("canny1", cv2.Canny(img1, 22, 66)) 

#等待显示
cv2.waitKey(0)
cv2.destroyAllWindows()

