import numpy as np
import cv2 as cv
import random

"""
三通道
"""
n_img = cv.imread("lenna.png")
w, h, c = n_img.shape[:3]
num = int(w * h * 1)
for i in range(c):
    for j in range(num):
        y = random.randint(0, h - 1)
        x = random.randint(0, w - 1)
        n_img[y, x][i] = n_img[y, x][i] + random.gauss(4, 10)
        if n_img[y, x][i] < 0:
            n_img[y, x][i] = 0
        if n_img[y, x][i] > 255:
            n_img[y, x][i] = 255

"""
单通道
"""
# img = cv.imread("lenna.png")
# n_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# w, h = n_img.shape[:2]
# num = int(w * h * 0.8)
# for j in range(num):
#     y = random.randint(0, h - 1)
#     x = random.randint(0, w - 1)
#     n_img[y, x] = n_img[y, x] + random.gauss(2, 4)
#     if n_img[y, x] < 0:
#         n_img[y, x] = 0
#     if n_img[y, x] > 255:
#         n_img[y, x] = 255

# cv.imshow("source", cv.cvtColor(cv.imread("lenna.png"), cv.COLOR_BGR2GRAY))

cv.imshow("source", cv.imread("lenna.png"))
cv.imshow("gauss_noise", n_img)
cv.waitKey(0)
