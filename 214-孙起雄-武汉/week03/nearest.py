# 实现最邻近插值
import cv2
import numpy as np


def zljcz(img):
    h, w, c = img.shape
    print("heigh=%s width=%s chanel=%s" % (h, w, c))
    emp = np.zeros((800, 800, c), np.uint8)
    print(emp)
    sh = 800/h
    sw = 800/w
    for i in range(800):
        for j in range(800):
            # 找出目标图像对应原图像最近的点
            x = int(i/sh)
            y = int(j/sw)
            emp[i, j] = img[x, y]
    return emp


img = cv2.imread("7.jpg")
zoom = zljcz(img)
# 打印图像矩阵信息
print(zoom)
print("---------------------------")
# 打印图像信息
print(zoom.shape)
print("---------------------------")
# 显示原图像
cv2.imshow("image", img)
# 显示放大后的图像
cv2.imshow("nearst", zoom)
cv2.waitKey(0)

