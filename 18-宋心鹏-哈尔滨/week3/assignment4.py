#单通道直方图均衡化，三通道仅需将RGB分开即可（注意拆分split后要用merge函数合并）
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#RGB转灰度图

dst = cv2.equalizeHist(gray)#直接equalizeHist调用函数，将灰度图像均衡化

hist = cv2.calcHist([dst],[0],None,[256],[0,256])#calcHist函数，计算输出图像直方图

plt.figure()
plt.hist(dst.ravel(), 256)
plt.show()

cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
cv2.waitKey(0)
