import cv2
import numpy as np
from matplotlib import pyplot as plt

# 获取灰度图
img = cv2.imread("lenna.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("image gray", img_gray)

# 直方图均衡化、生成直方图
dst = cv2.equalizeHist(img_gray)
hist = cv2.calcHist([dst], [0], None, [256], [0, 256])

# dst.ravel是将dst的二维数据降维成一维数组用于显示
plt.figure()
plt.hist(dst.ravel(), 256)
plt.show()

cv2.imshow("Histogram Equalization", np.hstack([img_gray, hist]))
cv2.waitKey(0)

# 彩色图直方图均衡化，要将图片分成三个通道cv2.split()，分别进行均衡化cv2.equalizeHist，再合并cv2.merge
img = cv2.imread("lenna.png")
cv2.imshow("src", img)

(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
result = cv2.merge((bH, gH, rH))
cv2.imshow("dst_rgb", result)
cv2.waitKey(0)
