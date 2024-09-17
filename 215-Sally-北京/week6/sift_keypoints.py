# sift算法，圈图像出关键点

import cv2
import numpy as np

img = cv2.imread('./lenna.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# sift = cv2.xfeatures2d.SIFT_create() # 专利算法，需将opencv版本退到3.4.2
sift = cv2.SIFT_create() # 新版函数
keypoints, descriptor = sift.detectAndCompute(gray, None)

img = cv2.drawKeypoints(
    image=img,
    outImage=img,
    keypoints=keypoints,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, # 对图像的每个关键点都绘制圆圈和方向
    color=(51, 163, 236)
)

# img = cv2.drawKeypoints(gray,keypoints,img) # 灰度图特征点圈出

cv2.imshow('sift_keypoints', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
