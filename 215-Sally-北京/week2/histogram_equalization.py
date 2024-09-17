# -*- coding: utf-8 -*-

import cv2
import numpy as np
from matplotlib import pyplot as plt

# 直方图均衡化
# 处理后图片感觉更清晰了

# 单通道
def histogramEqualizationGray(img):
    # 获取灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("image_gray", gray)
    
    # 灰度图像直方图均衡化
    dst = cv2.equalizeHist(gray)
    
    # 直方图
    hist = cv2.calcHist([dst],[0],None,[256],[0,256])
    
    plt.figure()
    plt.hist(dst.ravel(), 256)
    plt.show()
    
    cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
    cv2.waitKey(0)

# 三通道
def histogramEqualizationColorful(img):
    cv2.imshow("src", img)
    (b, g, r) = cv2.split(img)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)

    # 合并每一个通道
    result = cv2.merge((bH, gH, rH))
    cv2.imshow("dst_bgr", result)
    cv2.waitKey(0)



if __name__ == '__main__':
    # histogramEqualizationGray(cv2.imread('./lenna.png'))
    histogramEqualizationColorful(cv2.imread('./lenna.png'))
    
