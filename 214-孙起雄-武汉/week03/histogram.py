# 实现直方图均衡化

import cv2
import matplotlib.pyplot as plt

# 读取图像
img = cv2.imread('7.jpg', 0)

# 直方图均衡化
equ = cv2.equalizeHist(img)

# 绘制图像和直方图
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')

plt.subplot(2, 2, 2)
plt.hist(img.ravel(), 256)
plt.title('Original Histogram')

plt.subplot(2, 2, 3)
plt.imshow(equ, cmap='gray')
plt.title('Equalized Image')

plt.subplot(2, 2, 4)
plt.hist(equ.ravel(), 256)
plt.title('Equalized Histogram')

plt.show()


