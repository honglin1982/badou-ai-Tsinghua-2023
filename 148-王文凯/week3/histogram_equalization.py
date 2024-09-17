import cv2
import numpy as np
from matplotlib import pyplot as plt

# def histogram_equalization(img):
#     h, w = img.shape[:2]
#     dst_img = np.zeros((h, w, 0), img.dtype)
#     for y in range(h):
#         for x in range(w):
#             dst_img[y, x] = (img[y, x] / (h * w)) * 256 - 1
#     return dst_img

img = cv2.imread("./img/lenna.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# img_gray_hist_self = histogram_equalization(img_gray)
img_gray_hist = cv2.equalizeHist(img_gray)

# img_hist_self = histogram_equalization(img)
b, g, r = cv2.split(img)
b_hist, g_hist, r_hist = cv2.equalizeHist(b), cv2.equalizeHist(g), cv2.equalizeHist(r)
img_hist = cv2.merge((b_hist, g_hist, r_hist))

# cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
#  图像 通道 图像掩膜(与图像大小相同的8bit灰度图像矩阵) 每个图像维度参与直方图计算的直方图组数 参与直方图计算的每个维度的数值范围
# hist_self = cv2.calcHist(img_gray_hist_self, [0], None, [256], [0, 256])
hist = cv2.calcHist(img_gray_hist, [0], None, [256], [0, 256])

plt.figure()
plt.subplot(421)
plt.imshow(img)
plt.title("img")
plt.subplot(422)
plt.imshow(img_gray)
plt.title("img_gray")
plt.subplot(423)
plt.hist(img.ravel(), 256)
plt.title("img_hist")
plt.subplot(424)
plt.hist(img_gray.ravel(), 256)
plt.title("img_gray_hist")
plt.subplot(425)
plt.imshow(img_hist)
plt.title("img_hist")
plt.subplot(426)
plt.imshow(img_gray_hist)
plt.title("img_gray_hist")
plt.subplot(427)
plt.hist(img_hist.ravel(), 256)
plt.title("img_hist")
plt.subplot(428)
plt.hist(img_gray_hist.ravel(), 256)
plt.title("img_gray_hist")

plt.show()

cv2.imshow("color Histogram Equalization", np.hstack([img, img_hist]))
cv2.imshow("gray Histogram Equalization", np.hstack([img_gray, img_gray_hist]))
cv2.waitKey(0)