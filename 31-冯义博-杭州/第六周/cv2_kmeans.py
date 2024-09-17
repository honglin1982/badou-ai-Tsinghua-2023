import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread("lenna.png")
g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = g_img.shape[:2]
data = g_img.reshape((h * w, 1))
data = np.float32(data)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

flags = cv2.KMEANS_RANDOM_CENTERS
compactness, labels, centers = cv2.kmeans(data, 2, None, criteria, 10, flags)
dst = labels.reshape((h, w))

plt.figure(1)
plt.imshow(g_img, cmap='gray')


plt.figure(2)
plt.imshow(dst, cmap='gray')

plt.show()


