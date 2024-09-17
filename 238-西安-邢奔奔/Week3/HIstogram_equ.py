#!/usr/bin/python
# -*- coding:utf-8 -*-


import cv2
import numpy as np
from matplotlib import pyplot as plt

'''
equalizeHist(src,dst=None)
dst:默认即可
'''
img = cv2.imread('/Users/aragaki/artificial/image/lenna.png', 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

dst = cv2.equalizeHist(gray)

hist = cv2.calcHist([dst], [0], None, [256], [0, 255])

plt.figure("Hist")
plt.hist(dst.ravel(), 256)
plt.show()
# arrayForMyself = np.hstack([gray,dst])
cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
cv2.waitKey(0)
'''
cv2.destroyAllWindows()
cv2.waitKey(1)
'''
img = cv2.imread('/Users/aragaki/artificial/image/lenna.png', 1)
cv2.imshow("src", img)

(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
resu = cv2.merge((bH, gH, rH))
cv2.imshow("dst_rgb", resu)
cv2.waitKey(0)
