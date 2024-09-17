import cv2
import numpy as np
from matplotlib import pyplot as plt

'''
equalizeHist——直方图均衡化
函数原型:equalizeHist(src, dst=None)
src:图像矩阵（单通道图像）
dst:默认即可
'''

# #获取灰度图像
# img = cv2.imread("lenna.png", 1)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# #cv2.imshow("image_gray", gray)

# # 灰度图像直方图均衡化
# dst = cv2.equalizeHist(gray)

# #直方图
# hist = cv2.calcHist([dst], [0], None, [256], [0, 256])

# plt.figure()   #​用于创建一个新的图形窗口
# plt.hist(dst.ravel(), 256)   #绘制经过处理后的图像 ​dst​的直方图；​​dst.ravel()​将二维数组 ​dst​转换为一维数组。这是因为 ​plt.hist()​函数需要接受一维的数据作为输入，plt.hist()​函数用于绘制直方图。它接受一个一维数组作为输入，并根据数组中的数据绘制直方图，56​表示直方图的bin数目，即将灰度级别分成256个区间
# plt.show()   #显示图形窗口

# cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
# cv2.waitKey(0)



# 彩色图像直方图均衡化
img = cv2.imread("lenna.png", 1)
cv2.imshow("src", img)

# 彩色图像直方图均衡化需要分解通道，对每一个通道均衡化
(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
# 合并每一个通道
result = cv2.merge((bH, gH, rH))
cv2.imshow("dst_rgb", result)

cv2.waitKey(0)
