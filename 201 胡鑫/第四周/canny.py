import cv2
import numpy as np

'''
cv2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient ]]])
必要参数:
image: 需要处理的原图像, 该图像必须为单通道的灰度图
threshold1: 滞后阈值1
threshold2: 滞后阈值2
'''
img = cv2.imread('../lenna.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_c = cv2.Canny(gray, 200, 300)
cv2.imshow('canny', img_c)
cv2.waitKey(0)
cv2.destroyAllWindows()