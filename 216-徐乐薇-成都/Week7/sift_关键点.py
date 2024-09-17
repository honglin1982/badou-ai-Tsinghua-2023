import cv2
import numpy as np

img = cv2.imread('lenna.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create() # xfeature2d需要安装 opencv-contrib-python 库。这个库包含了 OpenCV 的许多扩展模块，例如 SIFT、SURF、ORB 等特征提取算法
kp, descriptors = sift.detectAndCompute(gray, None)  # 检测关键点和描述符 descriptors = None 表示不使用掩码

# 绘制关键点
img = cv2.drawKeypoints(image=img, outImage=img, keypoints=kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=(51, 163, 236)) # 绘制关键点 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS 表示绘制关键点的方向
# img = cv2.drawKeypoints(gray, kp, img)

cv2.imshow('sift_keypoints', img)
cv2.waitKey(0)