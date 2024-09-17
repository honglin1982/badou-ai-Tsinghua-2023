import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("lenna.png",1)                 #读图：imread使用了特殊用法
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)     #BGR转灰度图

dst = cv2.equalizeHist(gray)                    #equalizeHist（src#输入图像,dst#输出图像=None#默认没有）  直方图均衡化原型MOD；然后将灰度图传入

hist = cv2.calcHist([dst],[0],None,[256],[0,256])       #把传入的图像进行计算加工后输出图像的直方图

plt.figure()
plt.hist(dst.ravel(),256)
plt.show()

cv2.imshow("Histogram Equalization",np.hstack([gray,dst]))
cv2.waitKey(0)