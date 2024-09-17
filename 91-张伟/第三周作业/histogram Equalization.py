# 直方图
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 获取图像
img = cv2.imread("lenna.png",1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 直方图均衡化
dst = cv2.equalizeHist(gray)
b,g,r = cv2.split(img)
bh = cv2.equalizeHist(b)
gh = cv2.equalizeHist(g)
rh = cv2.equalizeHist(r)

# 直方图
hist = cv2.calcHist([dst], [0], None, [256], [0, 256])
result = cv2.merge((bh,gh,rh))


plt.figure()
plt.hist(dst.ravel(), 256)
plt.show()

cv2.imshow("hist", np.hstack([gray, dst]))#灰度图展示
cv2.imshow("hist_rgb",np.hstack([img, result]))#彩色图展示
cv2.waitKey()
