"""灰度图像直方图化"""


"""cv2.calcHist()函数的作用：通过直方图可以很好的对整幅图像的灰度分布有一个整体的了解
直方图的x轴是灰度值（0~255），y轴是图片中具有同一个灰度值的点的数目。而calcHist（）函数则可以帮助我们统计一幅图像的直方图
img = cv2.imread("/Users/songxinran/Documents/GitHub/badou-AI-Tsinghua-2023/lenna.png")
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
plt.hist(img.ravel(), 256)
plt.show()

hist是一个shape为(256,1)的数组，表示0-255每个像素值对应的像素个数，下标即为相应的像素值
plot一般需要输入x,y,若只输入一个参数，那么默认x为range(n)，n为y的长度
plt.plot(hist)
plt.show()
使用多个图像
hist = cv2.calcHist([img1,img2],[0,0],None,[256,256],[0,255,0,255])"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("/Users/songxinran/Documents/GitHub/badou-AI-Tsinghua-2023/lenna.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 显示一下灰度化后的图像
# cv2.imshow("image gray", gray)
# cv2.waitKey()
# 灰度图像直方图均衡化
dst = cv2.equalizeHist(gray)
# 画出直方图并显示,hist是一个shape为(256,1)的数组，表示0-255每个像素值对应的像素个数，下标即为相应的像素值
hist = cv2.calcHist([dst], [0], None, [256], [0, 256])
# plt.figure()
plt.hist(dst.ravel(), 256) # 将数组多维度拉成一维数组
plt.show()
# 显示直方图均衡化后的图片
# np.vstack()按垂直方向堆叠数组构成一个新的数组,堆叠的数组需要具有相同的维度
# np.hstack()按水平方向堆叠数组构成一个新的数组，堆叠的数组需要具有相同的维度
cv2.imshow("histogram equalization", np.vstack([gray, dst]))
cv2.waitKey()

