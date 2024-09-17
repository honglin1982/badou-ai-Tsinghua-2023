import cv2
import numpy as np

img = cv2.imread('../lenna.png')
img1 = img.copy()

# 任取四个点
src = np.float32([[154, 9], [477, 110], [27, 329], [362, 433]])
dst = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

# 求透视变换矩阵
m = cv2.getPerspectiveTransform(src, dst)
print('warpMatrx: ', m, sep='\n')

# 透视变换
img2 = cv2.warpPerspective(img1, m, (300, 300))
cv2.imshow('src', img)
cv2.imshow('dst', img2)
cv2.waitKey()
cv2.destroyAllWindows()