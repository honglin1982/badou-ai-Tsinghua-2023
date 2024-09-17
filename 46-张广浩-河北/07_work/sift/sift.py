
# -*- coding: utf-8 -*-
# @Time    : 2023/6/16 8:20
# @Author  : zgh
# @FileName: sift.py
# @Software: PyCharm

import cv2
import numpy as np

img = cv2.imread("../img/photo.jpg", 0)
sift = cv2.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(img, None)

res_img = cv2.drawKeypoints(image=img, outImage=img, keypoints=kp,
                        color=(153,204,255))
cv2.imshow('kp_img',res_img)
cv2.waitKey(0)
cv2.imwrite('../img/result.jpg', res_img)