import cv2
import numpy as np

# 最邻近插值
def fun(img):
    he, wi, channels = img.shape
    he1 = int(he * 1.5)
    wi1 = int(wi * 1.5)
    emptyimg = np.zeros((he1, wi1, channels), np.uint8)
    sh = he1 / he
    sw = wi1 / wi
    for i in range(he1):
        for j in range(wi1):
            x = int(i / sh + 0.5)
            y = int(j / sw + 0.5)
            emptyimg[i, j] = img[x, y]
    return emptyimg


img = cv2.imread("lenna.png")

zoom = fun(img)
cv2.imshow("lenna2",zoom)
cv2.waitKey()

