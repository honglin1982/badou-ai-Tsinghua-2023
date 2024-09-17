import numpy as np
import matplotlib.pyplot as plt
#用于调用rgb转灰度图的接口
from skimage.color import rgb2gray
import cv2

#读入图片
img = cv2.imread("lenna.png")
#可以直接读取灰度图像
#img = cv2.imread("lenna.png", 0)
#取得高度和宽度
height, width = img.shape[:2]
#定义灰度图像所需容器
imgGray = np.zeros([height, width], img.dtype)
#完成三通道图像到灰度单通道图像的复制
for i in range(height):
    for j in range(width):
        p = img[i,j]
        imgGray[i, j] = (p[0] * 11 + p[1] * 59 + p[2] * 30) / 100

print(imgGray)
print("image show gray : %s"%imgGray)
cv2.imshow("image show gray", imgGray)
#延迟等待
#cv2.waitKey(0)



plt.subplot(221)
#实现归一化
img = plt.imread("lenna.png")
plt.imshow(img)
print("----image lenna----")
print(img)


#调用实现灰度化
plt.subplot(222)
imgGray = rgb2gray(img)
plt.imshow(imgGray, cmap='gray')
print("----image Gray----")
print(imgGray)

#二值化
height, width = imgGray.shape
imgBinary = np.zeros([height, width], img.dtype)
for i in range(height):
    for j in range(width):
        imgBinary[i, j] = 1 if imgGray[i, j] >= 0.5 else 0

print("----imageBinary----")
print(imgBinary)
print(imgBinary.shape)

#调用实现二值化
#imgBinary = np.where(imgGray >= 0.5, 1, 0)

plt.subplot(223)
plt.imshow(imgBinary, cmap='binary')
plt.show()

