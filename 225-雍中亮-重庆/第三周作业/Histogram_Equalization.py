import cv2
import numpy as np
from matplotlib import pyplot as plt
# 彩色图像直方图均衡化
img = cv2.imread("lenna.png", 1)
# 1、表示彩色方式读取，0表示单通道方式读取
cv2.imshow("src", img)
# 彩色图像均衡化，需要分解通道，对每一个通道均衡化
(b,g,r)=cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
# 合并每一个通道
result = cv2.merge((bH,gH,rH))
cv2.imshow("dst_rgb",result)


# 获取灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow("image_gray", gray)

# 灰度图像直方图均衡化
dst = cv2.equalizeHist(gray)

# 直方图
hist = cv2.calcHist([dst],[0],None,[256],[0,256])
old_hist = cv2.calcHist([img],[0],None,[256],[0,256])

# 在同一张图中显示hist和old_hist
plt.figure()
plt.subplot(2, 1, 1)
plt.hist(dst.ravel(), 256)
plt.title('Equalized Histogram')
plt.subplot(2, 1, 2)
plt.hist(img.ravel(), 256)
plt.title('Original Histogram')
plt.show()

cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))


cv2.waitKey(0)