import cv2
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import numpy

def nearestInterpolation(img, h=800, w=800):

     sh, sw, channel = img.shape
     res = np.zeros((h, w, channel), np.uint8)
     scaleY, scaleX =  float(sh) / h,  float(sw) / w
     for i in range(h):
         for j in range(w):
            #寻找(i,j) 对应在原图像中 最近的点
            try: res[i, j] = img[int(i * scaleY + 0.5), int(j * scaleX + 0.5)]
            except : print("放大的长度或宽度等于或超过原来的二倍")
     return res

#机器码加速
@jit(nopython=True)
def biliInterpolation(img, h=1024, w = 1024):

    sh, sw, channel = img.shape
    res = np.zeros((h, w, channel), np.uint8)
    scaleY, scaleX = float(sh) / h, float(sw) / w
    for i in range(channel):
        for dy in range(h):
            for dx in range(w):

                #寻找 (dx, dy) 对应在原图像的 点 (x,y)
                sx, sy = (dx + 0.5) * scaleX - 0.5,  (dy + 0.5) * scaleY - 0.5
                #找出四个点
                sx0, sy0 = int(np.floor(sx)), int(np.floor(sy))
                sx1, sy1 = min(sx0 + 1, sw - 1), min(sy0 + 1, sh - 1)

                # (x0, y0) 和 (x1, y0) 的单线性插值
                r1 = (sx1 - sx) * img[sy0, sx0, i] + (sx - sx0) * img[sy0, sx1, i]
                # (x0, y1) 和 (x1, y1) 的单线性插值
                r2 = (sx1 - sx) * img[sy1, sx0, i] + (sx - sx0) * img[sy1, sx1, i]
                res[dy, dx, i] = int((sy1 - sy) * r1 + (sy - sy0) * r2)

    return res

#灰度图像直方图均衡化
def HisgramEqualizationGray(gray):

    return cv2.equalizeHist(gray)
#彩色图像直方图均衡化
def HisgramEqulizationRGB(img):

    b, g, r = cv2.split(img)
    bh, gh, rh = cv2.equalizeHist(b), cv2.equalizeHist(g), cv2.equalizeHist(r)
    return cv2.merge((bh, gh, rh))

if __name__ == '__main__':

    img = cv2.imread("lenna.png")
    nearestInter = nearestInterpolation(img)
    cv2.imshow("nearestInterpolation ", nearestInter)

    biliInter = biliInterpolation(img)
    cv2.imshow("biliInterpolation ", biliInter)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayHis = HisgramEqualizationGray(gray)
    his = cv2.calcHist([grayHis],[0], None, [256], [0, 256])
    plt.figure()
    plt.hist(grayHis.ravel(), 256)
    plt.show()
    cv2.imshow("HisgramEqualizationGray ", np.hstack([gray, grayHis]))

    rgbHis = HisgramEqulizationRGB(img)
    cv2.imshow("HisgramEqulizationRGB ", np.hstack([img, rgbHis]))

    cv2.waitKey()