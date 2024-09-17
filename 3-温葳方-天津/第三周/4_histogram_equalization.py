import cv2
import numpy as np
from matplotlib import pyplot as plt

color_image = cv2.imread("lenna.png", flags=cv2.IMREAD_COLOR)
gray_image = cv2.cvtColor(src=color_image, code=cv2.COLOR_BGR2GRAY)
# print(gray_image)
# cv2.imshow("gray_image", gray_image)
# cv2.waitKey(0)

destination_image = cv2.equalizeHist(src=gray_image)

# computing the histogram of the first channel of the image
# histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
histogram = cv2.calcHist([destination_image], [0], None, [256], [0, 256])

# plot the above computed histogram
plt.plot(histogram, color='b')
plt.title('Image Histogram For First Channel')
plt.show()

cv2.imshow("Histogram Equalization", np.hstack([gray_image, destination_image]))
cv2.waitKey(0)

# 彩色图像直方图均衡化
color_image = cv2.imread("lenna.png", 1)
cv2.imshow("src", color_image)
cv2.waitKey(0)

# 彩色图像均衡化,需要分解通道 对每一个通道均衡化
(b, g, r) = cv2.split(color_image)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
# 合并每一个通道
result = cv2.merge((bH, gH, rH))

cv2.imshow("dst_rgb", result)
cv2.waitKey(0)
