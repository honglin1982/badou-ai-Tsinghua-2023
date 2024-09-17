import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread("../lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = gray.shape

"""
直方图均衡化
"""
# 统计每个像素点个数
n_map = dict()
for i in range(h):
    for j in range(w):
        if n_map.get(gray[i, j]) is None:
            n_map[gray[i, j]] = 1
        else:
            n_map[gray[i, j]] += 1

# 计算均衡化像素值
sum_pi = 0
pi_map = dict()
for i in range(h):
    for j in range(w):
        if pi_map.get(gray[i, j]) is None:
            sum_pi += float(n_map[gray[i, j]]) / (h * w)
            q = int(sum_pi * 256 - 1 + 0.5)
            pi_map[gray[i, j]] = q

# 填充
n_img = np.zeros((h, w), img.dtype)
for i in range(h):
    for j in range(w):
        n_img[i, j] = pi_map[gray[i, j]]

hist = cv2.calcHist([n_img], [0], None, [256], [0, 256])

# 直方图
# plt.figure()
plt.hist(n_img.ravel(), 256)
plt.show()

dst = cv2.equalizeHist(gray)

(b, g, r) = cv2.split(img)
bh = cv2.equalizeHist(b)
gh = cv2.equalizeHist(g)
rh = cv2.equalizeHist(r)

result = cv2.merge((bh, gh, rh))

cv2.imshow("Histogram Equalization", np.hstack([gray, n_img, dst]))
cv2.imshow("result", np.hstack([img, result]))
cv2.waitKey(0)
